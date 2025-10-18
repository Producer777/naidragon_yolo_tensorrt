#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

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
        std::cout << "Usage: " << argv[0] << " <engine_file> <video_file>" << std::endl;
        return -1;
    }

    std::string engineFile = argv[1];
    std::string videoFile = argv[2];

    // 载入 engine
    auto engineData = readEngineFile(engineFile);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

    // 输入输出 tensor 名称
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

    // 输出假设最大 1000 个 float
    size_t outputSize = 1000 * sizeof(float);

    // 分配显存
    void* inputDevice = nullptr;
    void* outputDevice = nullptr;
    cudaMalloc(&inputDevice, inputSize);
    cudaMalloc(&outputDevice, outputSize);

    // 视频读取
    cv::VideoCapture cap(videoFile);
    if (!cap.isOpened()) return -1;

    // 读取视频信息
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 输出视频保存路径
    std::string outputVideo = "../naidragon_captured.mp4";
    cv::VideoWriter writer(outputVideo, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        // ============================
        // 预处理：letterbox + RGB + 归一化
        // ============================
        cv::Mat img;
        float scale;
        int top, left;
        letterbox(frame, inputW, inputH, img, scale, top, left);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);  // 归一化到 [0,1]

        std::vector<float> inputData(inputC * inputH * inputW);
        // CHW 排列
        for (int c = 0; c < inputC; ++c)
            for (int h = 0; h < inputH; ++h)
                for (int w = 0; w < inputW; ++w)
                    inputData[c * inputH * inputW + h * inputW + w] = img.at<cv::Vec3f>(h, w)[c];

        cudaMemcpy(inputDevice, inputData.data(), inputSize, cudaMemcpyHostToDevice);

        context->setTensorAddress(inputName, inputDevice);
        context->setTensorAddress(outputName, outputDevice);

        context->enqueueV3(0);

        std::vector<float> outputData(outputSize / sizeof(float));
        cudaMemcpy(outputData.data(), outputDevice, outputSize, cudaMemcpyDeviceToHost);

        // 解析输出
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

        for (const auto& b : finalBoxes) {
            cv::rectangle(frame, cv::Point(int(b.x1), int(b.y1)), cv::Point(int(b.x2), int(b.y2)), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "naidragon:" + std::to_string(int(b.conf * 100)) + "%",
                        cv::Point(int(b.x1), int(b.y1) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        // 写入输出视频帧
        writer.write(frame);    

        cv::imshow("YOLOv8 TensorRT 10", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cudaFree(inputDevice);
    cudaFree(outputDevice);
    delete context;
    delete engine;
    delete runtime;
    return 0;
}
