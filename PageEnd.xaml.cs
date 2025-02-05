namespace MauiApp3;

public partial class PageEnd : ContentPage
{

    public PageEnd(FileResult photoGauche, FileResult photoDroite, FileResult photoAvant, FileResult photoArrier)
    {
        InitializeComponent();

        ImageGauche.Source = ImageSource.FromFile(photoGauche.FullPath);
        ImageDroite.Source = ImageSource.FromFile(photoDroite.FullPath);
        ImageAvant.Source = ImageSource.FromFile(photoAvant.FullPath);
        ImageArrier.Source = ImageSource.FromFile(photoArrier.FullPath);
    }
}
