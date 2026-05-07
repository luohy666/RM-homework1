#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    fs::path root = fs::current_path();
    while (!fs::exists(root / "CMakeLists.txt") && root.has_parent_path())
        root = root.parent_path();
    fs::current_path(root);

    std::string imgDir = "data/calib_images";
    fs::create_directories("config");

    cv::Size boardSize(8, 5);     // 9x6 方格对应 8x5 内角点
    float squareSize = 22.5f;     // 改成你实际测量的小格边长，单位 mm

    std::vector<cv::Point3f> obj;
    for (int y = 0; y < boardSize.height; y++)
        for (int x = 0; x < boardSize.width; x++)
            obj.emplace_back(x * squareSize, y * squareSize, 0);

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;

    int total = 0;

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

        total++;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        imageSize = img.size();

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray,
            boardSize,
            corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
        );

        if (!found) continue;

        cv::cornerSubPix(
            gray,
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                30,
                0.001
            )
        );

        imagePoints.push_back(corners);
        objectPoints.push_back(obj);
    }

    if (imagePoints.size() < 8) {
        std::cerr << "图片总数: " << total << std::endl;
        std::cerr << "有效标定图片太少: " << imagePoints.size() << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    double rms = cv::calibrateCamera(
        objectPoints,
        imagePoints,
        imageSize,
        cameraMatrix,
        distCoeffs,
        rvecs,
        tvecs
    );

    cv::FileStorage fs("config/camera_params.yaml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs << "rms" << rms;
    fs << "square_size_mm" << squareSize;
    fs.release();

    std::cout << "格子大小: " << squareSize << " mm" << std::endl;
    std::cout << "重投影误差: " << rms << " 像素" << std::endl;

    std::cout << std::endl;
    std::cout << "内参矩阵:" << std::endl;
    std::cout << cameraMatrix << std::endl;

    std::cout << std::endl;
    std::cout << "畸变系数 (k1,k2,p1,p2,k3):" << std::endl;
    std::cout << distCoeffs << std::endl;

    return 0;
}