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
}

void t1(ov::InferRequest lm, int64_t out_token, int64_t SPECIAL_EOS_TOKEN, size_t seq_len, TextStreamer text_streamer, ov::Tensor position_ids, float* logits, size_t vocab_size)
{
    int max_sequence_length = 100;
    constexpr size_t BATCH_SIZE = 1;
    auto start = std::chrono::system_clock::now();
    while (out_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length) {
        // std::cout << "infer 1\n";
        ++seq_len;
        lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, seq_len});
        std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), seq_len, 1);
        position_ids.data<int64_t>()[0] = int64_t(seq_len - 1);
        // auto current_time = std::chrono::system_clock::now();
        // auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(current_time).time_since_epoch().count();
        // std::cout << "t1 start: " << now << std::endl;
        lm.start_async();
        lm.wait();
        // lm.infer();
        text_streamer.put(out_token);
        // auto current_time1 = std::chrono::system_clock::now();
        // auto now1 = std::chrono::time_point_cast<std::chrono::milliseconds>(current_time1).time_since_epoch().count();
        // std::cout << "t1 end: " << now1 << std::endl;
        logits = lm.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }
    std::cout << "\nseq_len = " << seq_len << std::endl;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "\ninfer time: " << diff.count() << " s\n";
    text_streamer.end();
    lm.reset_state();
}

void t2(ov::InferRequest lm, int64_t out_token, int64_t SPECIAL_EOS_TOKEN, size_t seq_len, TextStreamer text_streamer, ov::Tensor position_ids, float* logits, size_t vocab_size)
{
    int max_sequence_length = 100;
    constexpr size_t BATCH_SIZE = 1;
    auto start = std::chrono::system_clock::now();
    while (out_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length) {
        // std::cout << "infer 2\n";
        ++seq_len;
        lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, seq_len});
        std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), seq_len, 1);
        position_ids.data<int64_t>()[0] = int64_t(seq_len - 1);
        // auto current_time = std::chrono::system_clock::now();
        // auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(current_time).time_since_epoch().count();
        // std::cout << "t2 start: " << now << std::endl;
        lm.start_async();
        lm.wait();
        // lm.infer();
        // text_streamer.put(out_token);
        // auto current_time1 = std::chrono::system_clock::now();
        // auto now1 = std::chrono::time_point_cast<std::chrono::milliseconds>(current_time1).time_since_epoch().count();
        // std::cout << "t2 end: " << now1 << std::endl;
        logits = lm.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }
    std::cout << "seq_len2 = " << seq_len << std::endl;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "\ninfer time: " << diff.count() << " s\n";
    // text_streamer.end();
    lm.reset_state();
}

int main(int argc, char* argv[]) try {
    if (argc != 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>' <DEVICE_NAME>");
    }
    // Compile models
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    //Read the tokenizer model information from the file to later get the runtime information
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");
    // tokenizer and detokenizer work on CPU only
    ov::CompiledModel tokenizer_compile_model = core.compile_model(tokenizer_model, "CPU");

    ov::InferRequest tokenizer = tokenizer_compile_model.create_infer_request();
    ov::InferRequest tokenizer2 = tokenizer_compile_model.create_infer_request();

    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);
    auto [input_ids2, attention_mask2] = tokenize(tokenizer2, argv[2]);

    const std::string device_name = argv[3];

    ov::CompiledModel detokenizer_compile_model = core.compile_model(
        std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU");
    ov::InferRequest detokenizer = detokenizer_compile_model.create_infer_request();
    ov::InferRequest detokenizer2 = detokenizer_compile_model.create_infer_request();
    // The model can be compiled for GPU as well
    ov::CompiledModel compile_model = core.compile_model(
        std::string{argv[1]} + "/openvino_model.xml", device_name, {{"DEVICE_BIND_BUFFER","YES"},{"NUM_STREAMS","1"}});
    ov::InferRequest lm = compile_model.create_infer_request();
    ov::InferRequest lm2 = compile_model.create_infer_request();
    auto seq_len = input_ids.get_size();
    auto seq_len2 = input_ids2.get_size();
    
    // Initialize inputs
    lm.set_tensor("input_ids", input_ids);
    lm.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + seq_len, 0);
    constexpr size_t BATCH_SIZE = 1;
    // Input values are persistent between inference calls.
    // That allows to set values, which aren't going to change, only once
    lm.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    lm.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    // Initialize inputs
    lm2.set_tensor("input_ids", input_ids2);
    lm2.set_tensor("attention_mask", attention_mask2);
    ov::Tensor position_ids2 = lm2.get_tensor("position_ids");
    position_ids2.set_shape(input_ids2.get_shape());
    std::iota(position_ids2.data<int64_t>(), position_ids2.data<int64_t>() + seq_len2, 0);
    // Input values are persistent between inference calls.
    // That allows to set values, which aren't going to change, only once
    lm2.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    lm2.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    lm.infer();
    lm2.infer();

    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    size_t vocab_size2 = lm2.get_tensor("logits").get_shape().back();

    float* logits = lm.get_tensor("logits").data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;
    float* logits2 = lm2.get_tensor("logits").data<float>() + (seq_len2 - 1) * vocab_size2;
    int64_t out_token2 = std::max_element(logits2, logits2 + vocab_size2) - logits2;


    lm.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    position_ids.set_shape({BATCH_SIZE, 1});

    lm2.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    position_ids2.set_shape({BATCH_SIZE, 1});

    TextStreamer text_streamer{std::move(detokenizer)};
    TextStreamer text_streamer2{std::move(detokenizer2)};

    // Get the runtime info from the tokenizer model that we read earlier
    auto rt_info = tokenizer_model->get_rt_info(); //Get the runtime info for the model
    int64_t SPECIAL_EOS_TOKEN;

    if (rt_info.count("eos_token_id") > 0) { //check if the runtime information has a valid EOS token ID
        SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();
    } else {
        throw std::runtime_error("EOS token ID not found in model's runtime information.");
    }
    std::thread th1(t1, lm, out_token, SPECIAL_EOS_TOKEN, seq_len, text_streamer, position_ids, logits, vocab_size);
    std::thread th2(t2, lm2, out_token2, SPECIAL_EOS_TOKEN, seq_len2, text_streamer2, position_ids2, logits2, vocab_size2);
    th1.join();
    th2.join();
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
