# time-series-analysis

## Финальное задание

[Папка с данными (до и после препроцессинга)](./final-task/data/)

[Ноутбук с EDA](./final-task/solution/report_1__eda.ipynb)

[Ноутбук с отчетом по решению задачи прогнозирования](./final-task/solution/report_2__solution.ipynb)

Код:

- [utils.py](./final-task/solution/utils.py) - вспомогательные функции
- [forecasting.py](./final-task/solution/forecasting.py) - основной класс с решением

Вспомогательные файлы:

- [item2acfpacf.json](./final-task/solution/item2acfpacf.json) - статзначимые лаги ACF и PACF для определения параметров моделей ARIMA
- [metrics.json](./final-task/solution/metrics.json) - метрики различных моделей/товаров/горизонтов прогнозирования
- [selling_items_types.json](./final-task/solution/selling_items_types.json) - типы товаров по кол-ву продаж (низкие/высокие продажи)
