import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
f_start = 9.5e9  # начальная частота, Гц
f_end = 10e9  # конечная частота, Гц
f_step = 5e6  # шаг частоты, Гц
t_symbol = 0.01  # длительность символа, сек
t_sample = 0.001  # период дискретизации, сек

# Генерация частот
frequencies = np.arange(f_start, f_end, f_step)

# Генерация времени
t = np.arange(0, 3500 * pow(10,-12), 7 * pow (10,-12))

# Генерация ЛЧМ-сигнала
x = np.zeros((len(t), len(frequencies)))
for i, f in enumerate(frequencies):
    x[:, i] = np.sin(2 * np.pi * f * t)

# Вывод графика
plt.plot(t, x)
plt.xlabel('Время, сек')
plt.ylabel('Амплитуда')
plt.title('ЛЧМ-сигнал')
plt.show()

# Быстрое преобразование Фурье
fourier = np.fft.fft(x)

# Построение спектра сигнала
freq_axis = np.fft.fftfreq(len(t), d=1.0/t_sample)
plt.plot(freq_axis, np.abs(fourier))
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.title('Спектр ЛЧМ-сигнала')
plt.show()

# Окно Хэмминга
win_hamming = np.hamming(500)
for i in range(100):
    fourier[:,i] = fourier[:,i] * win_hamming
# Вывод нового графика
plt.plot(freq_axis, fourier)
plt.title('Hamming')
plt.show()