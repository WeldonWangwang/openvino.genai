// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <thread>
namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};
}  // namespace

void infer_task(int thread_id,
                char* prompt,
                ov::CompiledModel tokenizer_compile_model,
                ov::CompiledModel detokenizer_compile_model,
                ov::CompiledModel compile_model,
                std::shared_ptr<ov::Model> tokenizer_model,
                bool enable_stream,
                int max_sequence_length,
                int num_iters) {
    constexpr size_t BATCH_SIZE = 1;
    ov::InferRequest tokenizer = tokenizer_compile_model.create_infer_request();
    ov::InferRequest detokenizer = detokenizer_compile_model.create_infer_request();
    ov::InferRequest lm = compile_model.create_infer_request();
    TextStreamer text_streamer{std::move(detokenizer)};
    for (int i = 0; i < num_iters + 1; i++) {
        std::string log_tag = "warmup:";
        if (i != 0) {
            log_tag = std::to_string(i);
        }
        auto [input_ids, attention_mask] = tokenize(tokenizer, prompt);
        std::vector<int64_t> out_tokens;
        auto seq_len = input_ids.get_size();
        int64_t input_len = seq_len;
        std::cout << log_tag << " - " << std::to_string(thread_id) << " input token size: " << input_len << std::endl;
        lm.set_tensor("input_ids", input_ids);
        lm.set_tensor("attention_mask", attention_mask);
        ov::Tensor position_ids = lm.get_tensor("position_ids");
        position_ids.set_shape(input_ids.get_shape());
        std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + seq_len, 0);
        // constexpr size_t BATCH_SIZE = 1;
        // Input values are persistent between inference calls.
        // That allows to set values, which aren't going to change, only once
        lm.get_tensor("beam_idx").set_shape({BATCH_SIZE});
        lm.get_tensor("beam_idx").data<int32_t>()[0] = 0;
        auto start = std::chrono::system_clock::now();
        lm.infer();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> first_token_time = end - start;
        std::cout << log_tag << " - " << std::to_string(thread_id) << " First token time: " << first_token_time.count()
                  << " s\n";
        size_t vocab_size = lm.get_tensor("logits").get_shape().back();
        float* logits = lm.get_tensor("logits").data<float>() + (seq_len - 1) * vocab_size;
        int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;
        out_tokens.emplace_back(out_token);
        lm.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
        position_ids.set_shape({BATCH_SIZE, 1});
        // Get the runtime info from the tokenizer model that we read earlier
        auto rt_info = tokenizer_model->get_rt_info();  // Get the runtime info for the model
        int64_t SPECIAL_EOS_TOKEN;
        // std::cout << "SPECIAL_EOS_TOKEN: " << SPECIAL_EOS_TOKEN << std::endl;
        if (rt_info.count("eos_token_id") > 0) {  // check if the runtime information has a valid EOS token ID
            SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();
        } else {
            throw std::runtime_error("EOS token ID not found in model's runtime information.");
        }
        start = std::chrono::system_clock::now();
        ++seq_len;
        while (out_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length + input_len) {
            lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, seq_len});
            std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), seq_len, 1);
            position_ids.data<int64_t>()[0] = int64_t(seq_len - 1);
            lm.start_async();
            lm.wait();
            logits = lm.get_tensor("logits").data<float>();
            out_token = std::max_element(logits, logits + vocab_size) - logits;
            out_tokens.emplace_back(out_token);
            ++seq_len;
        }
        std::cout << log_tag << " - " << std::to_string(thread_id) << " seq_len = " << seq_len - input_len << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << log_tag << " - " << std::to_string(thread_id) << " Rest Tokens take time: " << diff.count()
                  << " s\n";
        std::cout << log_tag << " - " << std::to_string(thread_id)
                  << " Average rest token latency(s): " << diff.count() / (seq_len - input_len - 1) << std::endl;
        std::cout << log_tag << " - " << std::to_string(thread_id)
                  << " Rest Token/s: " << (seq_len - input_len - 1) / diff.count() << std::endl;
        std::cout << log_tag << " - " << std::to_string(thread_id)
                  << " Infer time(s): " << first_token_time.count() + diff.count() << std::endl;
        if (enable_stream) {
            for (const auto& token : out_tokens) {
                text_streamer.put(token);
            }
        }
        text_streamer.end();
        lm.reset_state();
    }
}

int main(int argc, char* argv[]) try {
    if (argc != 7) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> '<PROMPT>' <DEVICE_NAME> <NUM_DEVICES> <MAX_OUTPUT_TOKEN> <NUM ITERS>");
    }

    const std::string device_name = argv[3];
    int num_thread = std::stoi(argv[4]);
    int max_output_token_nums = std::stoi(argv[5]);
    // Compile models
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    // Read the tokenizer model information from the file to later get the runtime information
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::CompiledModel tokenizer_compile_model = core.compile_model(tokenizer_model, "CPU");
    ov::CompiledModel detokenizer_compile_model =
        core.compile_model(std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU");
    ov::AnyMap config = {};
    if (device_name.find("MULTI") == 0) {
        config = {{"DEVICE_BIND_BUFFER", "YES"}, {"NUM_STREAMS", "1"}};
    } else if (device_name.find("HETERO") == 0) {
        config = {{"MODEL_DISTRIBUTION_POLICY", "PIPELINE_PARALLEL"}, {"PERFORMANCE_HINT", "THROUGHPUT"}};
    }
    ov::CompiledModel compile_model =
        core.compile_model(std::string{argv[1]} + "/openvino_model.xml", device_name, config);
    // std::vector<ov::InferRequest> tokenizer_infer_request;
    // std::vector<>
    std::vector<std::thread> threads;
    bool enable_stream = true;
    for (int i = 0; i < num_thread; i++) {
        threads.emplace_back(std::thread(infer_task,
                                         i,
                                         argv[2],
                                         tokenizer_compile_model,
                                         detokenizer_compile_model,
                                         compile_model,
                                         tokenizer_model,
                                         enable_stream,
                                         max_output_token_nums,
                                         std::stoi(argv[6])));
        enable_stream = false;
    }

    for (auto& t : threads) {
        t.join();
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
