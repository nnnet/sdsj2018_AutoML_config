# sdsj2018 _AutoML_config

### Конфигурование пайплайна AutoML
Произвольный граф вычислений    
с возможностью отдельной настройки параметров каждого нода    
и общих настроек пайплайна    
Все ноды имеют универсальный интерфейс.    
Новые нужно зарегистрировать в lib.nodes._node_map    

### Обернут вызов моделей
vw    
h2o    
lightgbm    
arima    
Ridge / LogisticRegression    
RandomForestRegressor / RandomForestClassifier    
RidgeCV / LogisticRegressionCV    
BayesianRidge / GaussianNB    

### Параллелизм    
Ветвление графа пайплайна (кастомно)    
Паралеллизм потоков внутри нода (см. https://github.com/nnnet/Parallelize-pandas.DataFrame-Pool)    

### Stack    
Стак данных и результатов преобразования моделей внутри пайплайна    

### Feature_selection    
lib.features.select_features    
lgb & boruta    

### Примеры конфигурации    
В файле main.py    
закомментированы ниже    
`if __name__ == '__main__':    
    main()`    
    
В последней строке примеры вызова из командной строки для контеста SDSJ2018    

Примеры начала pipeline    
(посм. граф можно  `AutoML(args.model_dir, params=params_autoML).pipeline_draw(view=True)`)    
![Alt text](images/pipeline_starts_.jpg?raw=true "pipeline_starts")

Примеры окончания pipeline    
![Alt text](images/pipeline_ends.jpg?raw=true "pipeline_ends")

Пример конфигруации pipeline    
![Alt text](images/pipeline_config_part.jpg?raw=true "pipeline_config_part")


