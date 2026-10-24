using CarPricePrediction;
using Microsoft.ML;

var mlContext = new MLContext(seed: 42);

const string path = "used_cars_1000.csv";

// 1. Load the data
var data = mlContext.Data.LoadFromTextFile<CarData>(
    path: path,
    hasHeader: true,
    separatorChar: ',');

// 2. Split into train / test sets
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 42);

// 3. Build the pipeline
var pipeline = mlContext.Transforms.Categorical
    //Encodes the string categorical columns 
    .OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: nameof(CarData.Make))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: nameof(CarData.Model)))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TransmissionEncoded", inputColumnName: nameof(CarData.Transmission)))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "FuelTypeEncoded", inputColumnName: nameof(CarData.FuelType)))

    // Concatenate all features into a single feature vector
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        "MakeEncoded",
        "ModelEncoded",
        "TransmissionEncoded",
        "FuelTypeEncoded",
        nameof(CarData.Year),
        nameof(CarData.Mileage),
        nameof(CarData.EngineSize),
        nameof(CarData.Horsepower),
        nameof(CarData.Doors)
    ))
    
    // Normalize the features
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
        
    // Add the regression trainer
    .Append(mlContext.Regression.Trainers.Sdca(
        labelColumnName: nameof(CarData.Price),
        featureColumnName: "Features"));

// 4. Train the model on the training set
var trainedModel = pipeline.Fit(split.TrainSet);

// 5. Evaluate on the test set
var predictions = trainedModel.Transform(split.TestSet);

var metrics = mlContext.Regression.Evaluate(
    predictions,
    labelColumnName: nameof(CarData.Price));

Console.WriteLine($"R²:  {metrics.RSquared:0.###}");
Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:0.##}");
Console.WriteLine($"RMSE:{metrics.RootMeanSquaredError:0.##}");