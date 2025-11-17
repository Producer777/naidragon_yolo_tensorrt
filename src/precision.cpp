#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>  // C++17
namespace fs = std::filesystem;

using namespace nvinfer1;

// Logger
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

// 读取 engine 文件
std::vector<char> readEngineFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file: " + engineFile);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

struct Box {
    float x1, y1, x2, y2, conf;
};

float iou(const Box& a, const Box& b) {
    float interX1 = std::max(a.x1, b.x1);
    float interY1 = std::max(a.y1, b.y1);
    float interX2 = std::min(a.x2, b.x2);
    float interY2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, interX2 - interX1);
    float h = std::max(0.0f, interY2 - interY1);
    float inter = w * h;
    float unionArea = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return inter / unionArea;
}

std::vector<Box> nms(const std::vector<Box>& boxes, float iouThresh) {
    std::vector<Box> result;
    std::vector<Box> sortedBoxes = boxes;
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), [](const Box& a, const Box& b) { return a.conf > b.conf; });
    std::vector<bool> suppressed(sortedBoxes.size(), false);
    for (size_t i = 0; i < sortedBoxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(sortedBoxes[i]);
        for (size_t j = i + 1; j < sortedBoxes.size(); ++j)
            if (iou(sortedBoxes[i], sortedBoxes[j]) > iouThresh)
                suppressed[j] = true;
    }
    return result;
}

cv::Mat letterbox(const cv::Mat& src, int targetW, int targetH, cv::Mat& dst, float& scale, int& top, int& left) {
    int w = src.cols;
    int h = src.rows;
    scale = std::min(float(targetW) / w, float(targetH) / h);
    int newW = int(w * scale);
    int newH = int(h * scale);
    cv::resize(src, dst, cv::Size(newW, newH));
    top = (targetH - newH) / 2;
    left = (targetW - newW) / 2;
    cv::copyMakeBorder(dst, dst, top, targetH - newH - top, left, targetW - newW - left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return dst;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <engine_file> <image_folder>" << std::endl;
        return -1;
    }

    std::string engineFile = argv[1];
    std::string imageFolder = argv[2];

    // 加载 engine
    auto engineData = readEngineFile(engineFile);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

    const char* inputName = "images";
    const char* outputName = "output0";
    Dims inputDims = engine->getTensorShape(inputName);
    int batchSize = 1;
    inputDims.d[0] = batchSize;
    int inputC = inputDims.d[1];
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];
    size_t inputSize = inputC * inputH * inputW * sizeof(float);
    context->setInputShape(inputName, inputDims);

    size_t outputSize = 1000 * sizeof(float);  // 输出缓冲区大小
    void* inputDevice = nullptr;
    void* outputDevice = nullptr;
    cudaMalloc(&inputDevice, inputSize);
    cudaMalloc(&outputDevice, outputSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 输出保存目录
    std::string resultDir = "./results_txt/";
    fs::create_directories(resultDir);

    double total_infer_time = 0.0;
    int image_count = 0;

    for (const auto& entry : fs::directory_iterator(imageFolder)) {
        if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png") continue;
        std::string imagePath = entry.path().string();
        cv::Mat frame = cv::imread(imagePath);
        if (frame.empty()) continue;

        // ============================
        // 前处理
        // ============================
        cv::Mat img;
        float scale;
        int top, left;
        letterbox(frame, inputW, inputH, img, scale, top, left);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);

        std::vector<float> inputData(inputC * inputH * inputW);
        for (int c = 0; c < inputC; ++c)
            for (int h = 0; h < inputH; ++h)
                for (int w = 0; w < inputW; ++w)
                    inputData[c * inputH * inputW + h * inputW + w] = img.at<cv::Vec3f>(h, w)[c];

        cudaMemcpy(inputDevice, inputData.data(), inputSize, cudaMemcpyHostToDevice);
        context->setTensorAddress(inputName, inputDevice);
        context->setTensorAddress(outputName, outputDevice);

        // ============================
        // 推理 + 计时
        // ============================
        auto start = std::chrono::high_resolution_clock::now();
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();
        double infer_time = std::chrono::duration<double, std::milli>(end - start).count();

        total_infer_time += infer_time;
        image_count++;

        std::cout << "Image " << image_count
                  << " (" << entry.path().filename() << ") | "
                  << "Inference time: " << infer_time << " ms | "
                  << "FPS: " << 1000.0 / infer_time << std::endl;

        // ============================
        // 后处理：NMS + 坐标映射
        // ============================
        std::vector<float> outputData(outputSize / sizeof(float));
        cudaMemcpy(outputData.data(), outputDevice, outputSize, cudaMemcpyDeviceToHost);

        std::vector<Box> boxes;
        for (size_t i = 0; i + 5 < outputData.size(); i += 6) {
            float conf = outputData[i + 4];
            if (conf < 0.3f) continue;
            Box b;
            b.x1 = (outputData[i + 0] - left) / scale;
            b.y1 = (outputData[i + 1] - top) / scale;
            b.x2 = (outputData[i + 2] - left) / scale;
            b.y2 = (outputData[i + 3] - top) / scale;
            b.conf = conf;
            boxes.push_back(b);
        }
        auto finalBoxes = nms(boxes, 0.5f);

        // ============================
        // 保存检测结果（YOLO格式）
        // ============================
        std::ofstream fout(resultDir + entry.path().stem().string() + ".txt");
        for (const auto& b : finalBoxes) {
            // 输出格式：class_id x_center y_center width height confidence
            float x_center = (b.x1 + b.x2) / 2.0f / frame.cols;
            float y_center = (b.y1 + b.y2) / 2.0f / frame.rows;
            float width = (b.x2 - b.x1) / frame.cols;
            float height = (b.y2 - b.y1) / frame.rows;
            fout << 0 << " " << x_center << " " << y_center << " "
                 << width << " " << height << " " << b.conf << "\n";
        }
        fout.close();
    }

    std::cout << "======================================" << std::endl;
    std::cout << "Processed " << image_count << " images." << std::endl;
    std::cout << "Average Inference Time: " << total_infer_time / image_count << " ms" << std::endl;
    std::cout << "Average FPS: " << 1000.0 / (total_infer_time / image_count) << std::endl;
    std::cout << "Results saved to: " << resultDir << std::endl;

    // 清理资源
    cudaStreamDestroy(stream);
    cudaFree(inputDevice);
    cudaFree(outputDevice);
    delete context;
    delete engine;
    delete runtime;
    return 0;
}