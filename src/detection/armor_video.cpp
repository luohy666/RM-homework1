#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

struct Light {
    cv::RotatedRect rect;
    cv::Point2f center;
    float length;
    float width;
    float angle;
};

struct Armor {
    Light left;
    Light right;
    cv::RotatedRect rect;
    double score;
};

float normalizeAngle(float angle) {
    while (angle >= 90.0f) angle -= 180.0f;
    while (angle < -90.0f) angle += 180.0f;
    return angle;
}

float angleDiff(float a, float b) {
    float d = std::abs(normalizeAngle(a - b));
    return d > 90.0f ? 180.0f - d : d;
}

fs::path findProjectRoot() {
    fs::path root = fs::current_path();

    while (!fs::exists(root / "CMakeLists.txt") && root.has_parent_path()) {
        fs::path parent = root.parent_path();
        if (parent == root) break;
        root = parent;
    }

    return root;
}

cv::Mat getRedMask(const cv::Mat& frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2;
    cv::inRange(hsv, cv::Scalar(0, 35, 55), cv::Scalar(30, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(155, 35, 55), cv::Scalar(180, 255, 255), mask2);

    cv::Mat hsvMask = mask1 | mask2;

    std::vector<cv::Mat> bgr;
    cv::split(frame, bgr);

    cv::Mat rgDiff, rbDiff;
    cv::subtract(bgr[2], bgr[1], rgDiff);
    cv::subtract(bgr[2], bgr[0], rbDiff);

    cv::Mat rgMask, rbMask, rMask;
    cv::threshold(rgDiff, rgMask, 12, 255, cv::THRESH_BINARY);
    cv::threshold(rbDiff, rbMask, 8, 255, cv::THRESH_BINARY);
    cv::threshold(bgr[2], rMask, 70, 255, cv::THRESH_BINARY);

    cv::Mat mask = hsvMask & rgMask & rbMask & rMask;

    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2))
    );

    cv::dilate(
        mask,
        mask,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5))
    );

    cv::Mat roi = cv::Mat::zeros(mask.size(), CV_8UC1);
    int y0 = static_cast<int>(mask.rows * 0.25);

    cv::rectangle(
        roi,
        cv::Rect(0, y0, mask.cols, mask.rows - y0),
        cv::Scalar(255),
        -1
    );

    return mask & roi;
}

bool isLight(const std::vector<cv::Point>& contour, Light& light, const cv::Size& size) {
    double area = cv::contourArea(contour);

    if (area < 4.0 || area > size.area() * 0.02) {
        return false;
    }

    cv::RotatedRect rect = cv::minAreaRect(contour);

    float w = rect.size.width;
    float h = rect.size.height;

    if (w < 1.0f || h < 3.0f) {
        return false;
    }

    float length = std::max(w, h);
    float width = std::min(w, h);

    if (width <= 0.1f) {
        return false;
    }

    float ratio = length / width;

    if (ratio < 1.3f || ratio > 20.0f) {
        return false;
    }

    if (length < 5.0f || length > size.height * 0.45f) {
        return false;
    }

    float angle = rect.angle;

    if (w > h) {
        angle += 90.0f;
    }

    angle = normalizeAngle(angle);

    if (std::abs(angle) > 50.0f) {
        return false;
    }

    if (rect.center.y < size.height * 0.25f) {
        return false;
    }

    light.rect = rect;
    light.center = rect.center;
    light.length = length;
    light.width = width;
    light.angle = angle;

    return true;
}

bool makeArmor(const Light& a, const Light& b, Armor& armor, const cv::Size& size) {
    Light left = a.center.x < b.center.x ? a : b;
    Light right = a.center.x < b.center.x ? b : a;

    float dx = right.center.x - left.center.x;
    float dy = std::abs(right.center.y - left.center.y);

    if (dx <= 0.0f) {
        return false;
    }

    float avgLength = (left.length + right.length) * 0.5f;
    float heightRatio = std::max(left.length, right.length) / std::min(left.length, right.length);
    float angleDifference = angleDiff(left.angle, right.angle);
    float xRatio = dx / avgLength;
    float armorAngle = std::abs(std::atan2(dy, dx) * 180.0f / CV_PI);

    if (heightRatio > 2.4f) return false;
    if (angleDifference > 28.0f) return false;
    if (dy > avgLength * 1.0f) return false;
    if (xRatio < 0.5f || xRatio > 7.0f) return false;
    if (armorAngle > 38.0f) return false;
    if (dx > size.width * 0.35f) return false;

    cv::Point2f center = (left.center + right.center) * 0.5f;

    if (center.y < size.height * 0.25f) {
        return false;
    }

    std::vector<cv::Point2f> points;
    cv::Point2f p[4];

    left.rect.points(p);
    for (int i = 0; i < 4; i++) {
        points.push_back(p[i]);
    }

    right.rect.points(p);
    for (int i = 0; i < 4; i++) {
        points.push_back(p[i]);
    }

    cv::Point2f expectedCenter(size.width * 0.5f, size.height * 0.58f);
    double centerDist = cv::norm(center - expectedCenter) / size.width;

    armor.left = left;
    armor.right = right;
    armor.rect = cv::minAreaRect(points);

    armor.score =
        angleDifference * 1.5 +
        std::abs(heightRatio - 1.0f) * 25.0 +
        dy / avgLength * 20.0 +
        std::abs(xRatio - 2.5f) * 8.0 +
        centerDist * 15.0;

    return true;
}

bool findBestArmor(const std::vector<Light>& lights, Armor& bestArmor, const cv::Size& size) {
    bool found = false;
    double bestScore = 1e9;

    for (size_t i = 0; i < lights.size(); i++) {
        for (size_t j = i + 1; j < lights.size(); j++) {
            Armor armor;

            if (makeArmor(lights[i], lights[j], armor, size) && armor.score < bestScore) {
                bestScore = armor.score;
                bestArmor = armor;
                found = true;
            }
        }
    }

    return found;
}

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rect, const cv::Scalar& color, int thickness) {
    cv::Point2f p[4];
    rect.points(p);

    for (int i = 0; i < 4; i++) {
        cv::line(image, p[i], p[(i + 1) % 4], color, thickness);
    }
}

void drawArmor(cv::Mat& frame, const Armor& armor) {
    drawRotatedRect(frame, armor.rect, cv::Scalar(0, 0, 255), 2);

    cv::Point2f center = (armor.left.center + armor.right.center) * 0.5f;

    cv::putText(
        frame,
        "armor",
        center + cv::Point2f(-35, -18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 0, 255),
        2
    );
}

void processFrame(cv::Mat& frame) {
    cv::Mat mask = getRedMask(frame);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Light> lights;

    for (const auto& contour : contours) {
        Light light;

        if (isLight(contour, light, frame.size())) {
            lights.push_back(light);
        }
    }

    Armor armor;

    if (findBestArmor(lights, armor, frame.size())) {
        drawArmor(frame, armor);
    }
}

int main() {
    fs::path root = findProjectRoot();
    fs::current_path(root);

    fs::create_directories(root / "results");

    fs::path inputVideo = root / "data/armor_images/armor.mp4";
    fs::path outputVideo = root / "results/armor_result.mp4";

    std::cout << "项目根目录: " << root << std::endl;
    std::cout << "输入视频: " << inputVideo << std::endl;
    std::cout << "输出视频: " << outputVideo << std::endl;

    if (!fs::exists(inputVideo)) {
        std::cerr << "输入视频不存在: " << inputVideo << std::endl;
        return -1;
    }

    cv::VideoCapture cap(inputVideo.string());

    if (!cap.isOpened()) {
        std::cerr << "无法打开输入视频: " << inputVideo << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    if (fps <= 0.0 || std::isnan(fps)) {
        fps = 30.0;
    }

    cv::VideoWriter writer(
        outputVideo.string(),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频: " << outputVideo << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frameCount = 0;

    while (cap.read(frame)) {
        if (frame.empty()) {
            continue;
        }

        processFrame(frame);
        writer.write(frame);
        frameCount++;
    }

    cap.release();
    writer.release();

    std::cout << "视频处理完成: " << outputVideo << std::endl;
    std::cout << "处理帧数: " << frameCount << std::endl;

    return 0;
}