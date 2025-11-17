## 识别奶龙(单目标)
# 软件版本
cuda:12.4
tensorrt:10.13.0
# 运行
已经训练好的yolo v8模型，导出为best.onnx

利用
/usr/src/tensorrt/bin/trtexec \
--onnx=best.onnx \
--saveEngine=best.engine \
--minShapes=images:1x3x640x640 \
--optShapes=images:1x3x640x640 \
--maxShapes=images:16x3x640x640
在终端将onnx转换为engine

创建logger类处理日志
```cpp
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;
```
读取engine文件
```cpp
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
```
定义 Box 结构体，用于保存检测框左上 (x1,y1)、右下 (x2,y2) 和置信度 conf
```cpp
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

```
简单nms筛框
```cpp
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
```
letterbox,用于比例缩放
```cpp
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
```
载入engine并反序列化
```cpp
auto engineData = readEngineFile(engineFile);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

```    
输入输出 tensor 名称（默认）
```cpp
const char* inputName = "images";
const char* outputName = "output0";
```
cuda显存分配
```cpp
void* inputDevice = nullptr;
    void* outputDevice = nullptr;
    cudaMalloc(&inputDevice, inputSize);
    cudaMalloc(&outputDevice, outputSize);

```
读取视频并保存结果
```cpp
    cv::VideoCapture cap(videoFile);
    if (!cap.isOpened()) return -1;

    // 读取视频信息
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 输出视频保存路径
    std::string outputVideo = "output_result.mp4";
    cv::VideoWriter writer(outputVideo, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer." << std::endl;
        return -1;
    }
```
预处理
```cpp
cv::Mat img;
        float scale;
        int top, left;
        letterbox(frame, inputW, inputH, img, scale, top, left);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);  // 归一化到 [0,1]
```
解析
```cpp
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
         auto finalBoxes = nms(boxes, 0.5f);//nms阈值0.5

```
绘制
```cpp
 for (const auto& b : finalBoxes) {
            cv::rectangle(frame, cv::Point(int(b.x1), int(b.y1)), cv::Point(int(b.x2), int(b.y2)), cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "naidragon:" + std::to_string(int(b.conf * 100)) + "%",
                        cv::Point(int(b.x1), int(b.y1) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow("YOLOv8 TensorRT 10", frame);
        if (cv::waitKey(1) == 'q') break;
```
释放资源并退出
```cpp
    cudaFree(inputDevice);
    cudaFree(outputDevice);
    delete context;
    delete engine;
    delete runtime;
```
运行时采用
./yolo_trt ../best.engine ../naidragon.mp4
# 运行结果
提供naidragon.mp4作为演示
输出结果为naidragon_captured.mp4