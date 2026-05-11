#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

struct StockData {
    std::vector<int> day;
    std::vector<double> price;
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

StockData readStockCSV(const fs::path& path) {
    StockData data;
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << path << std::endl;
        return data;
    }

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        for (char& c : line) {
            if (c == ',') c = ' ';
        }

        std::stringstream ss(line);
        int day;
        double price;

        if (ss >> day >> price) {
            data.day.push_back(day);
            data.price.push_back(price);
        }
    }

    return data;
}

void saveFilterCSV(
    const fs::path& path,
    const std::vector<int>& day,
    const std::vector<double>& raw,
    const std::vector<double>& filtered
) {
    std::ofstream out(path);
    out << "day,original,kalman\n";
    out << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < day.size(); i++) {
        out << day[i] << "," << raw[i] << "," << filtered[i] << "\n";
    }
}

void savePredictCSV(
    const fs::path& path,
    const std::vector<int>& day,
    const std::vector<double>& predict
) {
    std::ofstream out(path);
    out << "day,predict\n";
    out << std::fixed << std::setprecision(6);

    for (size_t i = 0; i < day.size(); i++) {
        out << day[i] << "," << predict[i] << "\n";
    }
}

int main() {
    fs::path root = findRoot();
    fs::current_path(root);

    fs::path dataPath = root / "kalman/data/stock_prices.csv";
    fs::path resultDir = root / "kalman/results";

    fs::create_directories(resultDir);

    StockData data = readStockCSV(dataPath);

    if (data.price.empty()) {
        std::cerr << "股票数据为空。" << std::endl;
        return -1;
    }

    double price = data.price[0];
    double trend = 0.0;

    if (data.price.size() >= 2) {
        trend = data.price[1] - data.price[0];
    }

    double P00 = 1.0;
    double P01 = 0.0;
    double P10 = 0.0;
    double P11 = 1.0;

    double Q_price = 0.05;
    double Q_trend = 0.01;
    double R = 2.0;

    std::vector<double> filtered;
    filtered.push_back(price);

    for (size_t i = 1; i < data.price.size(); i++) {
        double pricePredict = price + trend;
        double trendPredict = trend;

        double P00p = P00 + P01 + P10 + P11 + Q_price;
        double P01p = P01 + P11;
        double P10p = P10 + P11;
        double P11p = P11 + Q_trend;

        double error = data.price[i] - pricePredict;
        double S = P00p + R;

        double K0 = P00p / S;
        double K1 = P10p / S;

        price = pricePredict + K0 * error;
        trend = trendPredict + K1 * error;

        P00 = (1.0 - K0) * P00p;
        P01 = (1.0 - K0) * P01p;
        P10 = P10p - K1 * P00p;
        P11 = P11p - K1 * P01p;

        filtered.push_back(price);
    }

    int predictDays = 10;
    std::vector<int> futureDay;
    std::vector<double> futurePrice;

    int lastDay = data.day.back();

    for (int i = 1; i <= predictDays; i++) {
        price = price + trend;
        trend = trend * 0.95;

        futureDay.push_back(lastDay + i);
        futurePrice.push_back(price);
    }

    saveFilterCSV(
        resultDir / "stock_filter_result.csv",
        data.day,
        data.price,
        filtered
    );

    savePredictCSV(
        resultDir / "stock_predict_result.csv",
        futureDay,
        futurePrice
    );

    std::cout << "股票滤波结果已保存: kalman/results/stock_filter_result.csv" << std::endl;
    std::cout << "股票预测结果已保存: kalman/results/stock_predict_result.csv" << std::endl;

    std::cout << "\n未来 " << predictDays << " 天预测价格:" << std::endl;
    for (size_t i = 0; i < futureDay.size(); i++) {
        std::cout << "Day " << futureDay[i]
                  << ": " << futurePrice[i] << std::endl;
    }

    return 0;
}