#include <iostream>
#include "vector_add.hpp"
#include "brightness.hpp"

int main() {
    int choice;
    std::cout << "Выберите задачу:\n1 - Сложение векторов\n2 - Яркость изображения\n";
    std::cin >> choice;

    if (choice == 1) {
        runVectorAddition();
    } else if (choice == 2) {
        runBrightnessIncrease();
    } else {
        std::cout << "Неверный выбор\n";
    }

    return 0;
}
