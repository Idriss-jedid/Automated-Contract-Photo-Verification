namespace MauiApp3;

public partial class WelcomePage : ContentPage
{
    public WelcomePage()
    {
        InitializeComponent();
        StartTimer();
    }

    private async void StartTimer()
    {
        await Task.Delay(5000); // Wait for 10 seconds
        Application.Current.MainPage = new NavigationPage(new PredictionPage1());
    }
}
