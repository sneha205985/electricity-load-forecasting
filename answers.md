# Technical Assessment Questions

## 1. What would you change if you had to forecast for hundreds of thousands of meters at once?

For hundreds of thousands of smart meters, I would redesign this from a local script into a scalable forecasting pipeline. The data would need to be processed using distributed tools such as Spark or cloud-based batch processing, with automated data validation, feature storage, model monitoring, and batch inference.

I would also avoid training one completely separate model for every meter unless necessary. Instead, I would consider using global models trained across many meters, or grouped models based on customer type, location, usage pattern, or meter category. This would make the system easier to maintain and more scalable.

In production, I would also add automated retraining, logging, alerting for data quality issues, monitoring for model drift, and efficient storage for historical meter readings and generated forecasts.

## 2. Do you think a model like this is used in practice by utilities, or would something simpler win?

Utilities do use forecasting models in practice, especially for short-term demand forecasting, grid planning, energy procurement, and operational decision-making. However, simple baselines such as same-hour-yesterday or same-hour-last-week are often surprisingly strong because electricity consumption follows daily and weekly patterns.

In a real production setting, I would not choose a complex model automatically. I would first compare it against strong naive baselines. The best model would be the simplest one that is reliable, explainable, easy to maintain, and clearly improves accuracy. A more complex model is only justified if it gives a measurable improvement in performance and business value.
