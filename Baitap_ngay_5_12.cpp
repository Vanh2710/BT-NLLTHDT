#include <iostream>
#include <vector>
#include <fstream>

class Matrix {
public:
    std::vector<std::vector<int> > data;

    Matrix(int rows, int cols) : data(rows, std::vector<int>(cols, 0)) {}

    
    void input() {
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[0].size(); j++) {
                std::cin >> data[i][j];
            }
        }
    }

    
    Matrix operator+(const Matrix& other) {
        Matrix result(data.size(), data[0].size());
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[0].size(); j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    
    Matrix operator-(const Matrix& other) {
        Matrix result(data.size(), data[0].size());
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[0].size(); j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    
    Matrix operator*(const Matrix& other) {
        Matrix result(data.size(), other.data[0].size());
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < other.data[0].size(); j++) {
                for (int k = 0; k < data[0].size(); k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    
    void display() {
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[0].size(); j++) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

class Logger {
public:
    std::ofstream logFile;

    Logger(const std::string& filename) {
        logFile.open(filename.c_str());
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void log(const std::string& operation, const Matrix& matrix) {
        logFile << operation << ":\n";
        for (int i = 0; i < matrix.data.size(); i++) {
            for (int j = 0; j < matrix.data[0].size(); j++) {
                logFile << matrix.data[i][j] << " ";
            }
            logFile << "\n";
        }
    }
};

int main() {
    
    Matrix A(2, 2);
    Matrix B(2, 2);

    
    Logger logger("log.txt");

    
    std::cout << "Nhap gia tri cho ma tran A:\n";
    A.input();
    std::cout << "Nhap gia tri cho ma tran B:\n";
    B.input();

    // C?ng ma tr?n A và B
    Matrix C = A + B;
    logger.log("A + B", C);

    // Tr? ma tr?n B t? A
    Matrix D = A - B;
    logger.log("A - B", D);

    // Nhân ma tr?n A và B
    Matrix E = A * B;
    logger.log("A * B", E);

    return 0;
}
