
#include <iostream>
#include <iomanip>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>


using namespace std;
using namespace cv;

//Variables de uso global
Mat img;
Mat rgb_img;
Mat maskdensity;
Mat masknucleos;
//RNG rng;

double numeronucleos, numestromas = 0;

//Pasamos por argumento la direccion y el nombre de la imagen (biopsia) a analizar
int main(int argc, char** argv) {
    //Si no se pasa el argumento, error y cerramos programa
    if (argc != 2) {
        cout << "Intruduzca una imagen como argumento para ejecutar correctamente el programa\n" << "\n";
        return -1;
    }

    //Si no, leemos imagen y guardamos en img
    img = imread(argv[1]);

    //Image or file no valid 
    if (!img.data) {
        cout << "Nombre de imagen incorrecto" << "\n";
        return -1;
    }
    //-------------------------------------------------------------------------------------------------------------------------
    //Manejo de la imagen de entrada y preparaciones iniciales:
    //Mostramos por pantalla la imagen original
    namedWindow("Imagen original", CV_WINDOW_NORMAL);
    imshow("Imagen original", img);
    waitKey(0);

    //Imprimimos por pantalla el size de la imagen en pixeles y mm
    cout << "Tamanho de la imagen en pixeles: " << img.size() << endl;
    cout << "Tamanho de la imagen en mm: [" << static_cast<double>(img.size().width) / 2100 << " x " <<
        static_cast<double>(img.size().height) / 2100 << "]" << endl;
    maskdensity = Mat::zeros(img.size(), CV_8UC3);
    masknucleos = Mat::zeros(img.size(), CV_32FC3);

    cvtColor(masknucleos, masknucleos, COLOR_BGR2GRAY);
    masknucleos = cv::Mat::ones(masknucleos.size(), masknucleos.type()) * 255;
    masknucleos = cv::Mat::zeros(masknucleos.size(), masknucleos.type());


    //--------------------------------------------------------------------------------------------------------------------------
    //Separamos la imagen original y usamos sus componentes B y G para separar la informacion que necesitamos de cualquier fondo (sea o no de la biotsia): 
    //BGR channels:
    vector<Mat> BGR_planes;
    split(img, BGR_planes);

    //Creamos una imagen solo con los componentes B y G
    Mat dst, srcBG = BGR_planes[0] + BGR_planes[1];

    //Aplicamos threshold
    threshold(srcBG, dst, 0, 255, THRESH_OTSU);

    //Mostramos imagen
    namedWindow("Imagen en formato binario (Otsu)", CV_WINDOW_NORMAL);
    imshow("Imagen en formato binario (Otsu)", dst);
    waitKey(0);


    //---------------------------------------------------------------------------------------------------------------------------

    //Luego de tener una imagen binarizada, nos encargamos de encotrar mediante contornos la informacion para poder crear las mascaras:

    vector<vector<Point>> contours;
    dst = dst.reshape(1, img.size().height);
    dst.convertTo(dst, DataType<uchar>::type);
    findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> hull(contours.size());


    for (int l = 0; l < contours.size(); l++) {

        double area = contourArea(contours[l], false);
        convexHull(contours[l], hull[l], false, true);
        double hullarea = contourArea(hull[l], false);


        //if para encontrar nucleos:
        if (fabs(contourArea(contours[l])) < 500 && (area / hullarea) >= 0.80) {

            drawContours(maskdensity, contours, l, Scalar(255, 255, 255), FILLED);
            drawContours(masknucleos, contours, l, Scalar(255, 255, 255), FILLED);
            numeronucleos++;

        }

        //if para encontrar stromas:
        if (fabs(contourArea(contours[l])) > 500) {

            drawContours(maskdensity, contours, l, Scalar(127, 127, 127), CV_FILLED);
            numestromas++;

        }
    }


    //Sacamos por consola el número de nucleos y estromas encontrados en la entrada 
    cout << "Numero de nucleos: " << numeronucleos << "\n";
    //cout << "Numero de estromas: " << numestromas << "\n";
    //Calculamos el número de células por mm cuadrado:
    double cellpermm = numeronucleos / ((static_cast<double>(img.size().width) / 2100) * (static_cast<double>(img.size().height) / 2100));
    cout << "Celulas por mm^2: " << cellpermm << "\n";

    //Mostramos mascara con solo los núcleos:
    namedWindow("Mascara nucleos", CV_WINDOW_NORMAL);
    imshow("Mascara nucleos", masknucleos);
    //Mostramos mascara de tres niveles:
    namedWindow("Mascara tres niveles", CV_WINDOW_NORMAL);
    imshow("Mascara tres niveles", maskdensity);

    //--------------------------------------------------------------------------------------------------------------
    //Dibujamos mapa de densidad para ver concentración de cólulas:
    //Creamos un kernel eliptico:
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat densidad;

    dilate(masknucleos, masknucleos, kernel, Point(-1, -1), 6);
    masknucleos.convertTo(masknucleos, CV_8U, 10);
    distanceTransform(masknucleos, densidad, CV_DIST_C, 3);
    densidad.convertTo(densidad, CV_8U, 10);
    applyColorMap(densidad, densidad, cv::COLORMAP_HOT);
    namedWindow("Mapa de densidad", CV_WINDOW_NORMAL);
    imshow("Mapa de densidad", densidad);
    imshow("Mascara de nucleos", masknucleos);
    waitKey(0);


    //-------------------------------------------------------------------------------------------------------------


}