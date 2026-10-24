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

// 6. Create a prediction engine for single predictions
var predictionEngine = mlContext.Model.CreatePredictionEngine<CarData, PricePrediction>(trainedModel);

var sampleCar = new CarData
{
    Make = "Toyota",
    Model = "Camry",
    Year = 2019f,
    Mileage = 42000f,
    EngineSize = 2.5f,
    Horsepower = 203f,
    Transmission = "Automatic",
    FuelType = "Gasoline",
    Doors = 4f
};

var prediction = predictionEngine.Predict(sampleCar);

Console.WriteLine();
Console.WriteLine("=================================");
Console.WriteLine("       Prediction Input Data");
Console.WriteLine("=================================");
Console.WriteLine($"Make:         {sampleCar.Make}");
Console.WriteLine($"Model:        {sampleCar.Model}");
Console.WriteLine($"Year:         {sampleCar.Year}");
Console.WriteLine($"Mileage:      {sampleCar.Mileage}");
Console.WriteLine($"Engine Size:  {sampleCar.EngineSize}");
Console.WriteLine($"Horsepower:   {sampleCar.Horsepower}");
Console.WriteLine($"Transmission: {sampleCar.Transmission}");
Console.WriteLine($"Fuel Type:    {sampleCar.FuelType}");
Console.WriteLine($"Doors:        {sampleCar.Doors}");

Console.WriteLine();
Console.WriteLine("=================================");
Console.WriteLine("          Model Metrics");
Console.WriteLine("=================================");
Console.WriteLine($"MAE:  {metrics.MeanAbsoluteError:C2}");
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:C2}");
Console.WriteLine($"R²:   {metrics.RSquared:0.###}");

Console.WriteLine();
Console.WriteLine("=================================");
Console.WriteLine("        Prediction Result");
Console.WriteLine("=================================");
Console.WriteLine($"Predicted Price: {prediction.Score:C2}");