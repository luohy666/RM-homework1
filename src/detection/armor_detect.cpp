#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

struct Light {
    cv::RotatedRect box;
    cv::Point2f c;
    float len, wid, ang;
};

float normAng(float a) {
    while (a >= 90) a -= 180;
    while (a < -90) a += 180;
    return a;
}

float angDiff(float a, float b) {
    float d = std::abs(normAng(a - b));
    return d > 90 ? 180 - d : d;
}

void drawBox(cv::Mat& img, const cv::RotatedRect& box) {
    cv::Point2f p[4];
    box.points(p);

    for (int i = 0; i < 4; i++) {
        cv::line(img, p[i], p[(i + 1) % 4], cv::Scalar(0, 0, 255), 3);
    }
}

bool isImg(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);

    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp";
}

cv::Mat blueMask(const cv::Mat& src) {
    cv::Mat blur, hsv;
    cv::GaussianBlur(src, blur, cv::Size(3, 3), 0);
    cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

    cv::Mat hsvMask;
    cv::inRange(hsv, cv::Scalar(85, 25, 80), cv::Scalar(145, 255, 255), hsvMask);

    std::vector<cv::Mat> ch;
    cv::split(blur, ch);

    cv::Mat diff, diffMask, brightMask, mask;
    cv::subtract(ch[0], ch[2], diff);

    cv::threshold(diff, diffMask, 15, 255, cv::THRESH_BINARY);
    cv::threshold(ch[0], brightMask, 90, 255, cv::THRESH_BINARY);

    cv::bitwise_or(hsvMask, diffMask, mask);
    cv::bitwise_and(mask, brightMask, mask);

    cv::Mat k1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat k2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k2);

    return mask;
}

bool getLight(const std::vector<cv::Point>& contour, Light& l, const cv::Size& s) {
    double area = cv::contourArea(contour);

    if (area < 8 || area > s.area() * 0.015) return false;

    cv::RotatedRect r = cv::minAreaRect(contour);

    float w = r.size.width;
    float h = r.size.height;

    if (w < 2 || h < 2) return false;

    if (w > h) {
        l.len = w;
        l.wid = h;
        l.ang = r.angle;
    } else {
        l.len = h;
        l.wid = w;
        l.ang = r.angle + 90;
    }

    float ratio = l.len / l.wid;

    if (ratio < 1.5 || ratio > 18) return false;
    if (l.len < 8 || l.len > s.height * 0.25) return false;
    if (l.wid > s.width * 0.05) return false;

    l.box = r;
    l.c = r.center;
    l.ang = normAng(l.ang);

    return true;
}

bool findArmor(const std::vector<Light>& lights, cv::RotatedRect& armor) {
    double best = 1e9;
    int bi = -1;
    int bj = -1;

    for (int i = 0; i < static_cast<int>(lights.size()); i++) {
        for (int j = i + 1; j < static_cast<int>(lights.size()); j++) {
            const Light& a = lights[i];
            const Light& b = lights[j];

            float lenRatio = std::max(a.len, b.len) / std::min(a.len, b.len);
            if (lenRatio > 2.0) continue;

            float ad = angDiff(a.ang, b.ang);
            if (ad > 28) continue;

            cv::Point2f v = b.c - a.c;
            float dist = std::sqrt(v.x * v.x + v.y * v.y);

            float avgLen = (a.len + b.len) / 2.0f;
            float avgWid = (a.wid + b.wid) / 2.0f;

            if (dist < avgWid * 3.0 || dist > avgLen * 6.0) continue;

            float centerAng = normAng(std::atan2(v.y, v.x) * 180.0f / CV_PI);
            float verticalDiff = std::abs(90.0f - angDiff(a.ang, centerAng));

            if (verticalDiff > 35) continue;

            float armorRatio = dist / avgLen;
            if (armorRatio < 0.8 || armorRatio > 5.5) continue;

            double score = ad * 2.0
                         + verticalDiff * 1.5
                         + std::abs(lenRatio - 1.0) * 50.0
                         + std::abs(armorRatio - 2.2) * 15.0;

            if (score < best) {
                best = score;
                bi = i;
                bj = j;
            }
        }
    }

    if (bi < 0 || bj < 0) return false;

    std::vector<cv::Point2f> pts;
    cv::Point2f p[4];

    lights[bi].box.points(p);
    for (int i = 0; i < 4; i++) pts.push_back(p[i]);

    lights[bj].box.points(p);
    for (int i = 0; i < 4; i++) pts.push_back(p[i]);

    armor = cv::minAreaRect(pts);

    return true;
}

int main() {
    fs::path root = fs::current_path();

    while (!fs::exists(root / "CMakeLists.txt") && root.has_parent_path()) {
        root = root.parent_path();
    }

    fs::current_path(root);

    std::string imgDir = "data/armor_images";
    std::string outDir = "results";

    fs::create_directories(outDir);

    if (!fs::exists(imgDir)) {
        std::cerr << "未找到图片文件夹: " << imgDir << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        if (!entry.is_regular_file() || !isImg(entry.path())) continue;

        cv::Mat src = cv::imread(entry.path().string());
        if (src.empty()) continue;

        cv::Mat mask = blueMask(src);
        cv::Mat result = src.clone();

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Light> lights;

        for (const auto& c : contours) {
            Light l;
            if (getLight(c, l, src.size())) {
                lights.push_back(l);
            }
        }

        cv::RotatedRect armor;

        if (findArmor(lights, armor)) {
            drawBox(result, armor);
        }

        fs::path outPath = fs::path(outDir) / entry.path().filename();
        cv::imwrite(outPath.string(), result);
    }

    return 0;
}