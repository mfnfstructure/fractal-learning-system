
---

README.md

```markdown
# 🧠 Fractal Learning System

**Самообучающаяся архитектура с внутренним диалогом, мета-рефлексией и мульти-модельным ансамблированием**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.17+-orange.svg)](https://www.tensorflow.org/js)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES2020-blue.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)

---

## 📌 Оглавление

- [Возможности](#-возможности)
- [Архитектура](#-архитектура)
- [Демо](#-демо)
- [Установка](#-установка)
- [Использование](#-использование)
- [API](#-api)
- [Визуализация](#-визуализация)
- [Roadmap](#-roadmap)
- [Лицензия](#-лицензия)

---

## ✨ Возможности

### 🧬 Когнитивные механизмы

| Механизм | Описание |
|----------|----------|
| **Двойная скорость** | Fast Brain (интуиция) + Slow Brain (анализ) с адаптивным взвешиванием |
| **Внутренний голос** | Генерация гипотез, рефлексия, связывание мыслей |
| **Энергетическая осознанность** | Система знает свою "усталость" и экономит ресурсы |
| **Аттракторы** | Устойчивые паттерны, влияющие на принятие решений |
| **Мета-обучение** | Самонастройка LR, температуры, exploration rate |
| **Epistemic мониторинг** | Калибровка уверенности, осознание незнания |
| **Когнитивный диссонанс** | Обнаружение и разрешение противоречий |

### 🧪 Мульти-модельная лаборатория

- **Загрузка нескольких моделей** — сравнивайте разные архитектуры
- **Ансамблирование** — голосование моделей для повышения точности
- **Экспорт/импорт весов** — сохраняйте обученные модели
- **Real-time предсказания** — интерактивный тест любой модели

### 📊 Визуализация

- Loss / Accuracy графики (train vs validation)
- Матрица ошибок (confusion matrix)
- Сравнение точности моделей
- Распределение уверенности предсказаний

---

## 🏗 Архитектура

```

┌─────────────────────────────────────────────────────────────────┐
│                      FractalController                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Fast Brain │  │  Slow Brain │  │ EnergySystem│              │
│  │ Distillation│  │   Model     │  │  (метаболизм)│              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│  ┌──────┴────────────────┴────────────────┴──────┐              │
│  │              Internal Voice                    │              │
│  │  • Генерация мыслей                            │              │
│  │  • Рефлексия                                   │              │
│  │  • Связывание (linking)                        │              │
│  └────────────────────────────────────────────────┘              │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Attractors  │  │ Echo Buffer │  │SelfLearningDB│              │
│  │ (паттерны)  │  │ (контекст)  │  │  (память)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Model Lab (UI)                          │
│  • Менеджер моделей • Загрузка файлов • Ансамблирование          │
│  • Графики • Confusion Matrix • Real-time предсказания           │
└─────────────────────────────────────────────────────────────────┘

```

### Компоненты

| Компонент | Файл | Описание |
|-----------|------|----------|
| `FractalController` | `Fls3.js` | Главный контроллер, координация всех систем |
| `DistillationModel` | `Fls3.js` | Fast Brain — быстрая интуитивная модель |
| `EnergySystem` | `Fls3.js` | Управление энергией, costs, регенерация |
| `SelfLearningDB` | `Fls3.js` | Персистентное хранилище опыта (IndexedDB) |
| `InternalVoice` | `Fls3.js` | Генерация и рефлексия мыслей |
| `SemanticEmbedder` | `Fls3.js` | Векторные представления текста |
| `Multi-Model Lab` | `flsPanel15.html` | Веб-интерфейс для экспериментов |

---


### Основные экраны

| Вкладка | Функции |
|---------|---------|
| **Валидация** | Обучение активной модели, тестирование, random baseline |
| **Предиктор** | Интерактивные предсказания с визуализацией уверенности |
| **Статистика** | Глобальная статистика, лучшая модель, сравнение |

---

## 📦 Установка

### Требования

- Node.js 18+ (для серверного запуска)
- Или любой современный браузер (Chrome, Firefox, Edge)

### Быстрый старт

```bash
# Клонирование репозитория

# Установка зависимостей (опционально, если используете npm)
npm install

# Запуск локального сервера
npx http-server -p 8080

# Открыть в браузере
open http://localhost:8080/flsPanel15.html
```

Браузерное использование

Просто откройте flsPanel15.html в браузере — все зависимости подтянутся через CDN.

---

🚀 Использование

1. Загрузка данных

Подготовьте JSON с обучающими данными:

```json
{
  "water": { "type": "liquid", "temp": 20 },
  "fire": { "type": "energy", "temp": 500 },
  "earth": { "type": "solid", "temp": 15 },
  "air": { "type": "gas", "temp": 22 }
}
```

2. Загрузка моделей

Загрузите несколько .js файлов с реализациями DistillationModel (или используйте встроенную).

3. Обучение

· Выберите активную модель
· Нажмите 🎓 Обучить активную
· Наблюдайте за прогрессом в реальном времени

4. Сравнение

· Обучите минимум 2 модели
· Нажмите 📊 Сравнить все модели
· Посмотрите, какая показала лучший результат

5. Ансамбль

· Нажмите 🎯 Ансамбль (голосование)
· Выберите модели для участия
· Запустите и сравните с отдельными моделями

---

📚 API

DistillationModel

```javascript
const model = new DistillationModel({
  inputDim: 32,
  hiddenUnits: [64, 32],
  dropoutRate: 0.2,
  learningRate: 0.001
});

// Обучение
await model.train(trainData, valData, 50, (epoch, total, logs) => {
  console.log(`Epoch ${epoch}/${total}: loss=${logs.loss}`);
});

// Предсказание
const predictions = await model.predict("water");
// [{ key: "water", confidence: 0.92 }, ...]

// Экспорт весов
const weights = model.exportWeights();

// Импорт весов
await model.importWeights(weights);
```

FractalController

```javascript
import { bootstrapFractalSystem } from './Fls3.js';

const { controller } = await bootstrapFractalSystem(model, {
  startInnerVoice: true,
  energy: { initial: 1.0, regenRate: 0.04 }
});

// Полный цикл обучения
const result = await controller.fullCycle("water", { key: "water" }, trainingData);

// Получение статистики
const stats = controller.getStats();
console.log(stats.energy, stats.attractors, stats.meta);
```

---

📊 Визуализация

График Что показывает
Loss Train Loss vs Validation Loss
Accuracy Train Acc vs Validation Acc vs Baseline
Confidence Распределение уверенности предсказаний
Comparison Сравнение точности всех моделей

Confusion Matrix

Автоматически строится после тестирования модели на валидационных данных. Зелёным подсвечены правильные предсказания (диагональ).

---

🗺 Roadmap

Ближайшее (v0.2)

· Сохранение/загрузка состояния всех моделей
· WebWorker для фонового обучения
· Экспорт графиков в PNG/SVG
· Drag-and-drop для порядка моделей в ансамбле

Среднесрочное (v0.3)

· ONNX экспорт/импорт
· Гибридные эмбеддинги (текст + структура)
· Мета-эволюция (система сама настраивает гиперпараметры)
· Интеграция с Transformers.js

Долгосрочное (v1.0)

· Распределённое обучение (WebRTC между вкладками)
· Мультимодальность (изображения + звук)
· Самостоятельное исследование окружения (active learning)

---

🤝 Контрибьюция

Приветствуются любые вклады!

1. Форкните репозиторий
2. Создайте ветку (git checkout -b feature/amazing-feature)
3. Зафиксируйте изменения (git commit -m 'Add amazing feature')
4. Запушьте (git push origin feature/amazing-feature)
5. Откройте Pull Request

Что особенно ценится

· Исправление утечек тензоров (TensorFlow.js memory leaks)
· Улучшение мета-обучения
· Новые стратегии для Internal Voice
· Документация и примеры

---

📄 Лицензия

Распространяется под лицензией MIT.
---

```


LICENSE (MIT)

```text
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

```
> ⚠️ **Экспериментальный прототип**  
> Это исследовательский проект, не готовый для продакшена. Возможны галлюцинации, утечки памяти, нестабильное поведение.
