using Microsoft.Maui;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SkiaSharp;

namespace MauiApp3;

public partial class PredictionPage2 : ContentPage
{
    private FileResult _selectedPhoto; // Field to store the selected photo from previous page
    private FileResult _photoGauche;
    private readonly InferenceSession _resnetSession;
    private readonly InferenceSession _vitSession;
    public PredictionPage2(FileResult photoGauche)
    {
        InitializeComponent();  // This links the XAML with the code-behind
        _photoGauche = photoGauche;
        ResultImage.Source = "cart.jpeg";

        string modelsDirectory = FileSystem.CacheDirectory;
        string resnetModelPath = Path.Combine(modelsDirectory, "modelresnet.onnx");
        string vitModelPath = Path.Combine(modelsDirectory, "best_vit_model33.onnx");

#if ANDROID
            // Extract raw resources to cache directory
            ExtractModelResource("modelresnet", resnetModelPath);
            ExtractModelResource("best_vit_model33", vitModelPath);
#endif

        _resnetSession = new InferenceSession(resnetModelPath);
        _vitSession = new InferenceSession(vitModelPath);
    }

#if ANDROID
        private void ExtractModelResource(string resourceName, string outputPath)
        {
            var context = Android.App.Application.Context;
            var resourceId = context.Resources.GetIdentifier(resourceName, "raw", context.PackageName);

            if (resourceId == 0)
            {
                throw new FileNotFoundException($"Resource '{resourceName}' not found.");
            }

            using var asset = context.Resources.OpenRawResource(resourceId);
            using var fileStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write);
            asset.CopyTo(fileStream);
        }
#endif

    private async void ProcessAndCheck(object sender, EventArgs e)
    {
        PredictionLabel.Text = "";
        NextButton.IsEnabled = false;
        RefreshButton.IsEnabled = false;
        var file = await PickOrCapturePhotoAsync();

        if (file != null)
        {
            _selectedPhoto = file; // Store the selected photo
            ResultImage.Source = ImageSource.FromFile(file.FullPath);

            // Get predictions from the ML.NET models
            var directionPrediction = PredictResnet(file.FullPath);
            var completePrediction = PredictVit(file.FullPath);

            // Check the predictions
            if (directionPrediction == "droite" && completePrediction == "1") // Assumes '1' means complete
            {
                PredictionLabel.Text = "Right direction, complete side.";
                NextButton.IsEnabled = true;
                RefreshButton.IsEnabled = true;
            }
            else if (directionPrediction == "droite")
            {
                PredictionLabel.Text = "Right direction, but not complete";
            }
            else
            {
                PredictionLabel.Text = "Direction is not right";
            }

        }
    }

    private async Task<FileResult> PickOrCapturePhotoAsync()
    {
        string action = await DisplayActionSheet("Choose Photo Source", "Cancel", null, "Take Photo", "Choose from Gallery");

        if (action == "Take Photo")
        {
            return await MediaPicker.CapturePhotoAsync();
        }
        else if (action == "Choose from Gallery")
        {
            return await MediaPicker.PickPhotoAsync();
        }

        return null;
    }

    private string PredictResnet(string imagePath)
    {
        // Preprocess the image
        var inputTensor = PreprocessImage(imagePath, 224, 224);

        // Run the model and get predictions
        var result = _resnetSession.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            });

        var prediction = result.First().AsEnumerable<float>().ToArray();
        int predictedIndex = Array.IndexOf(prediction, prediction.Max());

        var classNames = new[] { "autre", "avant", "arrier", "droite", "gauche" };
        return classNames[predictedIndex];
    }

    private string PredictVit(string imagePath)
    {
        // Preprocess the image
        var inputTensor = PreprocessImage(imagePath, 224, 224);

        // Run the model and get predictions
        var result = _vitSession.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            });

        var prediction = result.First().AsEnumerable<float>().ToArray();
        int predictedIndex = Array.IndexOf(prediction, prediction.Max());

        return predictedIndex.ToString(); // Assuming the model outputs 0 or 1 for completion
    }

    private Tensor<float> PreprocessImage(string imagePath, int width, int height)
    {
        using var image = SKBitmap.Decode(imagePath);
        using var resizedImage = image.Resize(new SKImageInfo(width, height), SKFilterQuality.Medium);
        var pixels = resizedImage.Pixels;

        var input = new DenseTensor<float>(new[] { 1, 3, height, width });
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var color = pixels[y * width + x];
                input[0, 0, y, x] = color.Red / 255.0f;
                input[0, 1, y, x] = color.Green / 255.0f;
                input[0, 2, y, x] = color.Blue / 255.0f;
            }
        }

        return input;
    }
    private async void NextPage(object sender, EventArgs e)
    {
        await Navigation.PushAsync(new PredictionPage3(_photoGauche,_selectedPhoto));
    }

    private void RefreshPage(object sender, EventArgs e)
    {
        ResultImage.Source = "cart.jpeg";
        PredictionLabel.Text = "";
        NextButton.IsEnabled = false;
        RefreshButton.IsEnabled = false;
    }

   
}
    
