# sdsj2018 _AutoML_config

### Конфигурование пайплайна AutoML
Произвольный граф вычислений    
с возможностью отдельной настройки параметров каждого нода    
и общих настроек пайплайна    

### Обернут вызов моделей
vw    
Ridge / LogisticRegression    
RandomForestRegressor / RandomForestClassifier    
RidgeCV / LogisticRegressionCV    
BayesianRidge / GaussianNB    
h2o (exclude_algos = ["GBM", "DeepLearning", "DRF"])    
	
### Параллелизм    
Ветвление графа пайплайна (кастомно)    
Паралеллизм потоков внутри нода (см. ф-ю _parallel_evolve)    

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
![Alt text](images/pipeline_starts_.jpg?raw=true "pipeline_starts")

Примеры окончания pipeline    
![Alt text](images/pipeline_ends.jpg?raw=true "pipeline_ends")

Пример конфигруации pipeline    
![Alt text](images/pipeline_config_part.jpg?raw=true "pipeline_config_part")


