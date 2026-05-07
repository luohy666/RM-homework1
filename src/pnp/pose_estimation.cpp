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

    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage file("config/camera_params.yaml", cv::FileStorage::READ);

    if (!file.isOpened()) {
        std::cerr << "未找到 config/camera_params.yaml，请先完成相机标定。" << std::endl;
        return -1;
    }

    file["camera_matrix"] >> cameraMatrix;
    file["dist_coeffs"] >> distCoeffs;

    float squareSize = 22.5f;
    if (!file["square_size_mm"].empty())
        file["square_size_mm"] >> squareSize;

    file.release();

    cv::Size boardSize(8, 5);   // 9x6 方格对应 8x5 内角点

    std::vector<cv::Point3f> objectPoints;
    for (int y = 0; y < boardSize.height; y++)
        for (int x = 0; x < boardSize.width; x++)
            objectPoints.emplace_back(x * squareSize, y * squareSize, 0);

    std::string imgDir = "data/calib_images";
    int count = 0;

    for (const auto& entry : fs::directory_iterator(imgDir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp")
            continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> imagePoints;

        bool found = cv::findChessboardCorners(
            gray,
            boardSize,
            imagePoints,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
        );

        if (!found) continue;

        cv::cornerSubPix(
            gray,
            imagePoints,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                30,
                0.001
            )
        );

        cv::Mat rvec, tvec, R;

        bool ok = cv::solvePnP(
            objectPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs,
            rvec,
            tvec
        );

        if (!ok) continue;

        cv::Rodrigues(rvec, R);

        std::cout << "\n=== " << entry.path().filename() << " ===" << std::endl;
        std::cout << "旋转矩阵 R:" << std::endl;
        std::cout << R << std::endl;
        std::cout << "平移向量 tvec:" << std::endl;
        std::cout << tvec << std::endl;

        count++;
    }

    std::cout << "\nPnP完成,有效图片数量: " << count << std::endl;

    return 0;
}