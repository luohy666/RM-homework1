#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;

struct Data {
    std::vector<double> x;
    std::vector<double> y;
};

fs::path findRoot() {
    fs::path p = fs::current_path();

    while (!fs::exists(p / "CMakeLists.txt") && p.has_parent_path()) {
        fs::path parent = p.parent_path();
        if (parent == p) break;
        p = parent;
    }

    return p;
}

Data readData(const fs::path& path) {
    Data data;
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << path << std::endl;
        return data;
    }

    std::string line;

    while (std::getline(file, line)) {
        for (char& c : line) {
            if (c == ',') c = ' ';
        }

        std::stringstream ss(line);
        double x, y;

        if (ss >> x >> y) {
            data.x.push_back(x);
            data.y.push_back(y);
        }
    }

    return data;
}

std::vector<double> kalmanFilter(const std::vector<double>& z, double Q, double R) {
    std::vector<double> filtered;

    if (z.empty()) {
        return filtered;
    }

    filtered.resize(z.size());

    double x = z[0];
    double P = 1.0;

    filtered[0] = x;

    for (size_t i = 1; i < z.size(); i++) {
        P = P + Q;

        double K = P / (P + R);

        x = x + K * (z[i] - x);

        P = (1.0 - K) * P;

        filtered[i] = x;
    }

    return filtered;
}

void saveCSV(
    const fs::path& path,
    const std::vector<double>& x,
    const std::vector<double>& raw,
    const std::vector<double>& filtered
) {
    std::ofstream out(path);

    if (!out.is_open()) {
        std::cerr << "无法保存文件: " << path << std::endl;
        return;
    }

    out << "x,original,kalman\n";
    out << std::fixed << std::setprecision(10);

    for (size_t i = 0; i < x.size(); i++) {
        out << x[i] << "," << raw[i] << "," << filtered[i] << "\n";
    }
}

void processOne(
    const fs::path& input,
    const fs::path& output,
    double Q,
    double R
) {
    Data data = readData(input);

    if (data.x.empty()) {
        std::cerr << "数据为空: " << input << std::endl;
        return;
    }

    std::vector<double> filtered = kalmanFilter(data.y, Q, R);

    saveCSV(output, data.x, data.y, filtered);

    std::cout << "已处理: " << input.filename()
              << " -> " << output.filename() << std::endl;
}

int main() {
    fs::path root = findRoot();
    fs::current_path(root);

    fs::path dataDir = root / "kalman/data";
    fs::path resultDir = root / "kalman/results";

    fs::create_directories(resultDir);

    processOne(dataDir / "homework_data_1.txt", resultDir / "homework_data_1_result.csv", 0.01, 0.20);
    processOne(dataDir / "homework_data_2.txt", resultDir / "homework_data_2_result.csv", 0.10, 0.80);
    processOne(dataDir / "homework_data_3.txt", resultDir / "homework_data_3_result.csv", 20.0, 80.0);
    processOne(dataDir / "homework_data_4.txt", resultDir / "homework_data_4_result.csv", 0.03, 0.20);

    std::cout << "作业一滤波完成，结果保存在 kalman/results。" << std::endl;

    return 0;
}