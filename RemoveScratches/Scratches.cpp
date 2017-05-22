#include "Scratches.hpp"

using namespace cv;
using namespace std;


//----------------FUNCION AUXILIAR ORDENAR VECTOR -------------------------------
bool sortcol( const vector<float>& v1, const vector<float>& v2 ) {
 return v1[4] > v2[4]; //en columna 6 guardo NFA
}

//----------------FUNCION AUXILIAR ORDENAR VECTOR -------------------------------
bool sortcol2( const vector<float>& v1, const vector<float>& v2 ) {
 return v1[4] < v2[4]; //en columna 6 guardo NFA
}

void SegmentIterator(const std::vector<std::vector<float> > SegmentsVector, const int index, const cv::Mat bin, vector<Point> &coordenadas, vector<int> &perfil)
{
	Point pt1,pt2;
	pt1.x=SegmentsVector[index][0];
	pt1.y=SegmentsVector[index][1];
	pt2.x=SegmentsVector[index][2];
	pt2.y=SegmentsVector[index][3];
	LineIterator it(bin, pt1, pt2, 8); // LineIterator Segment 1
	coordenadas.resize(it.count);
	perfil.resize(it.count); //creo buffer para guardar pixeles binarios.
	for(int j = 0; j < it.count; j++, ++it)
	{  //---------OBTENGO PERFIL DE RECTA Y SUS COORD DE PIXEL EN LA IMAGEN
		perfil[j] =  (int)bin.at<uchar>( it.pos() )/255; //Sustituir por MaxValue Natron
		coordenadas[j] = it.pos(); // guarda coord pixel en la imgen
	}

}

//----------------FUNCION AUXILIAR CALCULAR NFA -------------------------------
float NFA(const vector<int> perfil, const vector<cv::Point> coordenadas, const int c, const int f, const cv::Mat  PM , const long long int  Ntests )
{
    int l= f-c+1;; // l is the length of the segment considered
    double pm =0; //densidad promedio de todos los pixels de la recta
    double k0,r; //parametros cota de Hoeffding
    r=k0=0;
    float NFA =0; //Cota de Hoeffding.

    for (int ind = c; ind<f+1;ind++)
    { //hallo densidad promedio de Segment

        k0+=perfil[ind];
        pm+= PM.at<double>(coordenadas[ ind ]);
    }
    pm=pm/(double)l;
    r=(double)k0/(double)l;
    
    if (r==1)
    {
        NFA=exp(  -(l)*(r*log(r/pm))   ); // hallo cota de hoeffding, si es menor a cero el Segment es significativo
    }else{
        NFA=  exp(  -(l)* ( (r*log(r/pm)) + (   (1-r)*log((1-r)/(1-pm))) )   );
    }

    NFA=NFA*Ntests; //NFA

    return NFA;
}

//----------------BINARIZACION DE IMAGEN -------------------------------
void BinaryDetection(const cv::Mat src_bw, cv::Mat &bin, const int scratchWidth, const int medianDiffThreshold){
//int channels = src.channels();
        int nRows = src_bw.rows;
        int nCols = src_bw.cols;

        bin = Mat::zeros(nRows,nCols, CV_8U);
        Mat Ig;
        GaussianBlur(src_bw, Ig, Size(3,3), 1.0, 1.0,0  );
        //src_bw = Ig.clone();

        int s_med=medianDiffThreshold;
        int s_avg=20;
        int tmpPix_g;
        int tmpPix_m;

        //int scratchWidth=5;
        int R= round(scratchWidth/2); //Radio de la mediana
        int perfil [scratchWidth-1];
        float Avg_L;
        float Avg_R;
        int ind;

        for(int y = 0; y < nRows; y++){ //recorro imagen Src
            for ( int x = scratchWidth; x < nCols-scratchWidth; x++){

                    tmpPix_g= (int)Ig.at<uchar>(y,x);

                    ind=0;
                    for (int i = x-R; i<=x+R ; i++){
                        if (i!=x) {
                            perfil[ind]= (int)Ig.at<uchar>(y,i);
                            ind++;
                        }
                    }

                    std::sort(perfil, perfil+scratchWidth-1);
                    tmpPix_m=perfil[R-1];

                    //Promedios laterales

                    Avg_L=Avg_R=0;

                    for (int i=x-2*R-1; i<=x-R-1;i++){
                        Avg_L+=(int)Ig.at<uchar>(y,i);
                    }
                    Avg_L/=(float)(R+1);
                    for (int i=x+R+1; i<=x+2*R+1;i++){
                        Avg_R+=(int)Ig.at<uchar>(y,i);
                    }
                    Avg_R/=(float)(R+1);

                    //CONDICIONES BOOLEANAS PARA DETECCION BINARIA

                    //cout << "tmpPix_g = " << tmpPix_g << " y tmpPix_m = " << tmpPix_m << " -> RESTA = " << (abs(tmpPix_g-tmpPix_m)) <<  endl;
                    //cout << "Avg_L = " << Avg_L << " y Avg_R = " << Avg_R<< " -> RESTA = " << (fabs(Avg_L-Avg_R)) <<  endl;

                    if ( (abs(tmpPix_g-tmpPix_m)>s_med) && (fabs(Avg_L-Avg_R)<s_avg) ) {
                        bin.at<uchar>(y,x)=255;
                        //cout << "ENTRO Y PONGO 1" << endl;
                    }
                    else{
                        bin.at<uchar>(y,x)=0;
                        //cout << "ENTRO Y PONGO 0" << endl;
                    }


            }
        }//Fin de Binarizacion
        //imshow("Binarizacion",bin);
}

void HoughSpeedUp(const cv::Mat bin, const int thresholdHough,const int inclination, std::vector<std::vector<float> > &lines_Hough){

    vector<Vec2f> lines; 

    //HoughLines(InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn=0, double stn=0 )
    HoughLines(bin, lines, 1, CV_PI/180, thresholdHough);

    //float largo_seg;
    int VertMargin = 2*bin.cols; //useful for determine a segment outside of the image bounding


    for( size_t i = 0; i < lines.size(); i++ ) //ITERO PARA CADA LINEA
    {

        float theta = lines[i][1];

        if ((fabs(theta*180 / CV_PI)<=inclination)  )   // solo acepto rectas con inclinacion menor a "inclination" grados
        {
            float rho = lines[i][0];
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            Point pt1, pt2;
            pt1.x = cvRound(x0 + VertMargin*(-b));
            pt1.y = cvRound(y0 + VertMargin*(a)); 
            pt2.x = cvRound(x0 - VertMargin*(-b));
            pt2.y = cvRound(y0 - VertMargin*(a));

            vector<float> Segment;
            Segment.push_back( pt1.x );
            Segment.push_back( pt1.y );
            Segment.push_back( pt2.x );
            Segment.push_back( pt2.y );

            lines_Hough.push_back(Segment); // agrego nuevo Segment

        }



    }
}

void PixelDensity(cv::Mat bin,cv::Mat &PM){

    int nRows = bin.rows;
    int nCols = bin.cols;
    PM = Mat::zeros(nRows,nCols, CV_64FC1);

        //----------------MAPA DENSIDAD PIXELES IMAGEN -------------------------------
        //algunas constantes que utiliza el algoritmo
        //float minLength = round(nRows/10); //largo minimo aceptado para un scratch

        //long long int Ntests = (long long int)nRows*nRows*nCols*40; //Numero de tests para metodo a contrario
        //cout<< "pixels = "<< Ntests << endl;

        //CV_8UC1
        int L = (int)(nCols/30);
        float max_density=0;
        float density;
        int lim_y, lim_x, cant_pixels;

        for(int i = 0; i < nRows; ++i)
        {
            for ( int j = 0; j < nCols; ++j)
            {
                //int val = dst.at<uchar>(i,j);

                //Cuadrante 1 ----------------------------------
                lim_y=max(0,i-L+1);
                lim_x=max(0,j-L+1);
                cant_pixels=0;
                density=0;
                for (int y=lim_y;y<=i;y++){
                    for (int x=lim_x; x<=j; x++){
                        cant_pixels++;
                        density += (int)bin.at<uchar>(y,x)/255;
                    }
                }
                density = density/cant_pixels;
                //cout<< "pixels = "<< cant_pixels << endl;
                //cout<< "density = "<< density << endl;
                max_density=density;

                //Cuadrante 2 ----------------------------------
                lim_y=max(0,i-L+1);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                density=0;
                for (int y=lim_y;y<=i;y++){
                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)bin.at<uchar>(y,x)/255;
                    }
                }
                density = density/cant_pixels;
                //cout<< "pixels = "<< cant_pixels << endl;
                //cout<< "density = "<< density << endl;
                max_density=max(max_density,density);

                //Cuadrante 3 ----------------------------------
                lim_y=min(nRows,i+L);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                density=0;
                for (int y=i;y<=lim_y-1;y++){
                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)bin.at<uchar>(y,x)/255;
                    }
                }
                density = density/cant_pixels;
                //cout<< "pixels = "<< cant_pixels << endl;
                //cout<< "density = "<< density << endl;
                max_density=max(max_density,density);

                //Cuadrante 4 ----------------------------------
                lim_y=min(nRows,i+L);
                lim_x=max(0,j-L+1);
                cant_pixels=0;
                density=0;
                for (int y=i;y<=lim_y-1;y++){
                    for (int x=lim_x; x<=j; x++){
                        cant_pixels++;
                        density += (int)bin.at<uchar>(y,x)/255;
                    }
                }
                density = density/cant_pixels;
                //cout<< "pixels = "<< cant_pixels << endl;
                //cout<< "density = "<< density << endl;
                max_density=max(max_density,density);

                PM.at<double>(i,j)=(double)max_density;
            }
        }


        //imshow("Desnidad",PM);



}

void PixelDensity2(cv::Mat bin,cv::Mat &PM){

//----------------GENERA MAPA DENSIDAD PIXELES IMAGEN -------------------------------
    int nRows = bin.rows;
    int nCols = bin.cols;
    //float minLength = round(nRows/10); //largo minimo aceptado para un scratch

    //long long int Ntests = (long long int)nRows*nRows*nCols*40; //Numero de tests para metodo a contrario
    //cout<< "pixels = "<< Ntests << endl;

    //CV_8UC1
    int L = (int)(nCols/30);
    float max_density=0;
    float density;
    int lim_y, lim_x, cant_pixels;
    float column_sum=0;

    PM = Mat::zeros(nRows,nCols, CV_64FC1);
    cv::Mat PM_aux=Mat::zeros(nRows,nCols, CV_64FC1);

    for(int y = 0; y < nRows-L; ++y){
        for ( int x = 0; x < nCols; ++x){

            for (int i=0; i<L; i++){
                column_sum+=(int)bin.at<uchar>(y+i,x)/255;
            }
            PM_aux.at<double>(y,x)=(double)column_sum;

            column_sum=0;
        }
    }

    for(int i = 0; i < nRows; ++i)
        {
            for ( int j = 0; j < nCols; ++j)
            {
              
                //Cuadrante 1 ----------------------------------
                density=0;

                if ( (i-L+1<0) || (j-L+1<0 ) )  { // el cuadrado esta por fuera del borde

                    lim_y=max(0,i-L+1);
                    lim_x=max(0,j-L+1);
                    
                    cant_pixels=0;
                    for (int y=lim_y;y<=i;y++){

                        uchar *Pix = bin.ptr<uchar>(y);
                        Pix +=j-L; //corrijo el puntero en la posicion x

                        for (int x=lim_x; x<=j; x++){
                            cant_pixels++;
                            density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                            Pix++;
                        }
                    }
                    density = density/cant_pixels;
                    //cout<< "pixels = "<< cant_pixels << endl;
                    //cout<< "density = "<< density << endl;
                    max_density=density;

                } else {

                    
                    for (int x=0;x<=L;x++ ){
                        density += PM_aux.at<double>(i-L,j-L+x);
                    }
                    density=density/(L*L);
                    max_density=density;

                }

                
                //Cuadrante 2 ----------------------------------
                density=0;

                if ( (i-L+1<0) || (j+L>nCols ) )  { // el cuadrado esta por fuera del borde

                    lim_y=max(0,i-L+1);
                    lim_x=min(nCols,j+L);
                    cant_pixels=0;
                    for (int y=lim_y;y<=i;y++){

                        uchar *Pix = bin.ptr<uchar>(y);
                        Pix +=j; //corrijo el puntero en la posicion x

                        for (int x=j; x<=lim_x-1; x++){
                            cant_pixels++;
                            density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                            Pix++;
                        }
                    }
                    density = density/cant_pixels;
                    //cout<< "pixels = "<< cant_pixels << endl;
                    //cout<< "density = "<< density << endl;
                    max_density=max(max_density,density);

                } else {

                    for (int x=0;x<=L;x++ ){
                        density += PM_aux.at<double>(i-L,j+x);
                        
                    }
                    density=density/(L*L);
                    max_density=max(max_density,density);

                }
                
                
                //Cuadrante 3 ----------------------------------
                density=0;

                if ( (i+L>nRows) || (j+L>nCols ) )  { // el cuadrado esta por fuera del borde

                    lim_y=min(nRows,i+L);
                    lim_x=min(nCols,j+L);
                    cant_pixels=0;
                    for (int y=i;y<=lim_y-1;y++){

                        uchar *Pix = bin.ptr<uchar>(y);
                        Pix +=j; //corrijo el puntero en la posicion x

                        for (int x=j; x<=lim_x-1; x++){
                            cant_pixels++;
                            density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                            Pix++;
                        }
                    }
                    density = density/cant_pixels;
                    //cout<< "pixels = "<< cant_pixels << endl;
                    //cout<< "density = "<< density << endl;
                    max_density=max(max_density,density);

                } else {

                    

                    for (int x=0;x<=L;x++ ){
                        density += PM_aux.at<double>(i,j+x);
                        
                    }
                    density=density/(L*L);
                    max_density=max(max_density,density);

                }

                
                //Cuadrante 4 ----------------------------------
                density=0;

                if ( (i+L>nRows) || (j-L+1<0 ) )  { // el cuadrado esta por fuera del borde

                    lim_y=min(nRows,i+L);
                    lim_x=max(0,j-L+1);
                    cant_pixels=0;
                    for (int y=i;y<=lim_y-1;y++){

                        uchar *Pix = bin.ptr<uchar>(y);
                        Pix +=j-L; //corrijo el puntero en la posicion x


                        for (int x=lim_x; x<=j; x++){
                            cant_pixels++;
                            density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                            Pix++;
                        }
                    }
                    density = density/cant_pixels;

                    max_density=max(max_density,density);

                } else {
                    
                    for (int x=0;x<=L;x++ ){

                        density += PM_aux.at<double>(i,j-L+x);

                    }
                    density=density/(L*L);
                    max_density=max(max_density,density);
                
                }
                

                PM.at<double>(i,j)=(double)max_density;
            }
        }


        PM_aux.release();




}

void PixelDensity3(cv::Mat bin,cv::Mat &PM){
//----------------GENERA MAPA DENSIDAD PIXELES IMAGEN -------------------------------
    int nRows = bin.rows;
    int nCols = bin.cols;
    //float minLength = round(nRows/10); //largo minimo aceptado para un scratch

    //long long int Ntests = (long long int)nRows*nRows*nCols*40; //Numero de tests para metodo a contrario
    //cout<< "pixels = "<< Ntests << endl;

    //CV_8UC1
    int L = (int)(nCols/30);
    float max_density=0;
    float density;
    int lim_y, lim_x, cant_pixels;
    float column_sum=0;

    PM = Mat::zeros(nRows,nCols, CV_64FC1);
    cv::Mat PM_aux=Mat::zeros(nRows-L,nCols, CV_64FC1);

    for(int y = 0; y < nRows-L; ++y){
        for ( int x = 0; x < nCols; ++x){

            for (int i=0; i<L; i++){
                column_sum+=(int)bin.at<uchar>(y+i,x)/255;
            }
            PM_aux.at<double>(y,x)=(double)column_sum;

            column_sum=0;
        }
    }

    //ESTRATEGIA DE BORDE
    //BANDA INFERIOR
    for(int i = 0; i < L; ++i){
            for ( int j = 0; j < nCols; ++j){

                //cuadrante1
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=max(0,j-L+1);

                cant_pixels=0;
                for (int y=lim_y;y<=i;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j-L; //corrijo el puntero en la posicion x

                    for (int x=lim_x; x<=j; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=density;

                //Cuadrante 2
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=lim_y;y<=i;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);   

                //Cuadrante3
                density=0;
   
                lim_y=min(nRows,i+L);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density); 

                //Cuadrante 4 ----------------------------------
                density=0;

                lim_y=min(nRows,i+L);
                lim_x=max(0,j-L+1);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j-L; //corrijo el puntero en la posicion x


                    for (int x=lim_x; x<=j; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);

                PM.at<double>(i,j)=(double)max_density;

            }
    }

    //BANDA SUPERIOR
    for(int i = nRows-L-1; i < nRows; ++i){
            for ( int j = 0; j < nCols; ++j){

                //cuadrante1
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=max(0,j-L+1);

                cant_pixels=0;
                    for (int y=lim_y;y<=i;y++){

                        uchar *Pix = bin.ptr<uchar>(y);
                        Pix +=j-L; //corrijo el puntero en la posicion x

                        for (int x=lim_x; x<=j; x++){
                            cant_pixels++;
                            density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                            Pix++;
                        }
                    }
                    density = density/cant_pixels;
                    max_density=density;

                //Cuadrante 2
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=lim_y;y<=i;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);   

                //Cuadrante3
                density=0;
   
                lim_y=min(nRows,i+L);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density); 

                //Cuadrante 4 ----------------------------------
                density=0;

                lim_y=min(nRows,i+L);
                lim_x=max(0,j-L+1);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j-L; //corrijo el puntero en la posicion x


                    for (int x=lim_x; x<=j; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);

                PM.at<double>(i,j)=(double)max_density;

            }
    }

    //BANDA IZQ
    for(int i = L; i < nRows-L-1; ++i){
            for ( int j = 0; j < L; ++j){

                //cuadrante1
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=max(0,j-L+1);

                cant_pixels=0;
                    for (int y=lim_y;y<=i;y++){

                        uchar *Pix = bin.ptr<uchar>(y);
                        Pix +=j-L; //corrijo el puntero en la posicion x

                        for (int x=lim_x; x<=j; x++){
                            cant_pixels++;
                            density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                            Pix++;
                        }
                    }
                    density = density/cant_pixels;
                    max_density=density;

                //Cuadrante 2
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=lim_y;y<=i;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);   

                //Cuadrante3
                density=0;
   
                lim_y=min(nRows,i+L);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density); 

                //Cuadrante 4 ----------------------------------
                density=0;

                lim_y=min(nRows,i+L);
                lim_x=max(0,j-L+1);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j-L; //corrijo el puntero en la posicion x


                    for (int x=lim_x; x<=j; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);

                PM.at<double>(i,j)=(double)max_density;

            }
    }

    //BANDA DERECHA BORDE
    for(int i = L; i < nRows-L-1; ++i){
            for ( int j = nCols-L-1; j < nCols; ++j){

                //cuadrante1
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=max(0,j-L+1);

                cant_pixels=0;
                    for (int y=lim_y;y<=i;y++){

                        uchar *Pix = bin.ptr<uchar>(y);
                        Pix +=j-L; //corrijo el puntero en la posicion x

                        for (int x=lim_x; x<=j; x++){
                            cant_pixels++;
                            density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                            Pix++;
                        }
                    }
                    density = density/cant_pixels;
                    max_density=density;

                //Cuadrante 2
                density=0;
                lim_y=max(0,i-L+1);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=lim_y;y<=i;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);   

                //Cuadrante3
                density=0;
   
                lim_y=min(nRows,i+L);
                lim_x=min(nCols,j+L);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j; //corrijo el puntero en la posicion x

                    for (int x=j; x<=lim_x-1; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density); 

                //Cuadrante 4 ----------------------------------
                density=0;

                lim_y=min(nRows,i+L);
                lim_x=max(0,j-L+1);
                cant_pixels=0;
                for (int y=i;y<=lim_y-1;y++){

                    uchar *Pix = bin.ptr<uchar>(y);
                    Pix +=j-L; //corrijo el puntero en la posicion x


                    for (int x=lim_x; x<=j; x++){
                        cant_pixels++;
                        density += (int)*Pix/255;//(int)bin.at<uchar>(y,x)/255;
                        Pix++;
                    }
                }
                density = density/cant_pixels;
                max_density=max(max_density,density);

                PM.at<double>(i,j)=(double)max_density;

            }
    }

    //REGION CON OPTIMIZACION
    for(int i = L; i < nRows-L; ++i)
        {
            for ( int j = L; j < nCols-L; ++j)
            {
              
                //Cuadrante 1 ----------------------------------
                density=0;

                    double *Pix = PM_aux.ptr<double>(i-L);
                    Pix+=j-L; // corrijo puntero
                    for (int x=0;x<=L;x++ ){
                        density += *Pix;//PM_aux.at<double>(i-L,j-L+x);
                        Pix++;
                    }
                    density=density/(L*L);
                    max_density=density;

                
                //Cuadrante 2 ----------------------------------
                density=0;

                    double *Pix2 = PM_aux.ptr<double>(i-L);
                    Pix2+=j; // corrijo puntero

                    for (int x=0;x<=L;x++ ){
                        density += *Pix2;//PM_aux.at<double>(i-L,j+x);
                        Pix2++;
                    }
                    density=density/(L*L);
                    max_density=max(max_density,density);

            
                //Cuadrante 3 ----------------------------------
                density=0;

                    double *Pix3 = PM_aux.ptr<double>(i);
                    Pix3+=j; // corrijo puntero

                    for (int x=0;x<=L;x++ ){
                        density += *Pix3;//PM_aux.at<double>(i,j+x);
                        Pix3++;
                    }
                    density=density/(L*L);
                    max_density=max(max_density,density);
  
                //Cuadrante 4 ----------------------------------
                density=0;

                    double *Pix4 = PM_aux.ptr<double>(i);
                    Pix4+=j-L; // corrijo puntero

                    for (int x=0;x<=L;x++ ){
                        density += *Pix4;//PM_aux.at<double>(i,j-L+x);
                        Pix4++;
                    }
                    density=density/(L*L);
                    max_density=max(max_density,density);

                PM.at<double>(i,j)=(double)max_density;
            }
        }

}

void ExclusionPrinciple(std::vector<std::vector<float> > &Detecciones_EXC,std::vector<std::vector<float> > &Detecciones_EXCOUT, const cv::Mat bin, const cv::Mat PM, const int nfaThreshold, const long long int  Ntests,const int minLength, int minDistance ){

    sort(Detecciones_EXC.begin(), Detecciones_EXC.end(),sortcol2); //ORDENO NFA DESCENDENTE
 
    vector<int> intersecciones;
    bool FLAG_intersect=0;
    int cant_Segments=Detecciones_EXC.size();
    double EPS=pow(10.0,nfaThreshold);
    int distance;
    //minDistance=3;

    vector<bool> ExcludeSegment(Detecciones_EXC.size(),false);


    int i=1; //empieza en una porque el primero se queda si o si por tener menor NFA de todas

        while ( i<cant_Segments )
        {

            if (Detecciones_EXC.size() == 0)
                break;

            sort(Detecciones_EXC.begin()+i, Detecciones_EXC.end(),sortcol2);

            vector<int> perfil1; //Segment 1 ----
            vector<Point> coordenadas1;
            SegmentIterator(Detecciones_EXC, i, bin, coordenadas1, perfil1);

            vector<bool> Intersected(coordenadas1.size(),false);


            for (int j=i-1;j>=0;--j) {

                if (ExcludeSegment[j] != true)
                {
                    vector<int> perfil2; //Segment 2 ----
                    vector<Point> coordenadas2;
                    SegmentIterator(Detecciones_EXC, j, bin, coordenadas2, perfil2);



                    //RECORRO AMBOS Y BUSCO PIXELES REPETIDOS
                    bool cercanos=0;
                    //int distancia =0;
                    int cant_intersecciones=0;
                    for (size_t ii=0;ii<coordenadas1.size();ii++){

                        for (size_t jj=0;jj<coordenadas2.size();jj++){

                            distance=pow(coordenadas2[jj].x-coordenadas1[ii].x,2.0)+pow(coordenadas2[jj].y-coordenadas1[ii].y,2.0);
                            int d2=pow(minDistance,2);
                            if (( coordenadas1[ii]==coordenadas2[jj])||(distance<=d2 ))//abs(coordenadas1[ii].x- coordenadas2[jj].x)<=10)  /*&&( abs(coordenadas1[ii].y- coordenadas2[jj].y)<=1))*/

                            cercanos=1;

                            if(cercanos==1)
                            { //<--------------Agregar radio bola Thau_x
                                FLAG_intersect=1;
                                Intersected[ii]=true;
                                intersecciones.push_back(ii);
                                cant_intersecciones++;
                                cercanos=0;
                            }
                        }
                    }

                   if (cant_intersecciones !=0)
                    {
                       ExcludeSegment[i]=true;

                        int c,f;
                        c=f=-1;
                        float NFA_nueva;
                        int cant_Segments_nuevos = 0;

                        if ((Intersected[0]==true) && (Intersected[Intersected.size()-1]==true)){ //todos los pixeles son intersecciones
                            ExcludeSegment[i]=true;
                        }
                        else if (Intersected[0]==true)
                        {// la interseccion esta en el principio del scratch

                            f=Intersected.size()-1;
                            c=0;
                            while(Intersected[c]==true){
                                c++;
                            }

                            NFA_nueva= NFA(perfil1,coordenadas1,c,f,PM,Ntests);

                            if (( NFA_nueva<EPS) && (NFA_nueva!=0) && (f-c+1 > minDistance))
                            {//Segment SIGNIFICATIVO LO GUARDO EN UNA TABLA JUNTO A SU NFA
                                cant_Segments_nuevos++;

                                vector<float> Segment;
                                Segment.push_back( (coordenadas1[ c ].x) );
                                Segment.push_back( (coordenadas1[ c ].y) );
                                Segment.push_back( (coordenadas1[ f ].x)  );
                                Segment.push_back( (coordenadas1[ f ].y)  );
                                Segment.push_back( NFA_nueva );

                                Detecciones_EXC.push_back(Segment); // agrego nuevo Segment
                                ExcludeSegment.push_back(false); // agrego segmento no exlcuido
                            }

                        }else if(Intersected[Intersected.size()-1]==true){ //la interseccion esta al final del scratch
                            c=0;
                            f=0;
                            while(Intersected[f]==false){
                                f++;
                            }
                            f--;
                            NFA_nueva= NFA(perfil1,coordenadas1,c,f,PM,Ntests);

                            if (( NFA_nueva<EPS) && (NFA_nueva!=0) && (f-c+1 > minDistance))
                            {//Segment SIGNIFICATIVO LO GUARDO EN UNA TABLA JUNTO A SU NFA
                                cant_Segments_nuevos++;

                                vector<float> Segment;
                                Segment.push_back( (coordenadas1[ c ].x) );
                                Segment.push_back( (coordenadas1[ c ].y) );
                                Segment.push_back( (coordenadas1[ f ].x)  );
                                Segment.push_back( (coordenadas1[ f ].y)  );
                                Segment.push_back( NFA_nueva );

                                Detecciones_EXC.push_back(Segment); // agrego nuevo Segment
                                ExcludeSegment.push_back(false);
                            }

                        }else{//la interseccion esta en el medio del scratch
                            //primer tramo
                            c=0;
                            f=0;
                            while(Intersected[f]==false){
                                f++;
                            }
                            f--;
                            NFA_nueva= NFA(perfil1,coordenadas1,c,f,PM,Ntests);

                            if (( NFA_nueva<EPS) && (NFA_nueva!=0) && (f-c+1 > minDistance))
                            {//Segment SIGNIFICATIVO LO GUARDO EN UNA TABLA JUNTO A SU NFA
                                cant_Segments_nuevos++;

                                vector<float> Segment;
                                Segment.push_back( (coordenadas1[ c ].x) );
                                Segment.push_back( (coordenadas1[ c ].y) );
                                Segment.push_back( (coordenadas1[ f ].x)  );
                                Segment.push_back( (coordenadas1[ f ].y)  );
                                Segment.push_back( NFA_nueva );

                                Detecciones_EXC.push_back(Segment); // agrego nuevo Segment
                                ExcludeSegment.push_back(false);
                            }
                            c=f+1;
                            while(Intersected[c]==true){
                                c++;
                            }
                            NFA_nueva= NFA(perfil1,coordenadas1,c,f,PM,Ntests);

                            if (( NFA_nueva<EPS) && (NFA_nueva!=0) && (f-c+1 > minDistance))
                            {//Segment SIGNIFICATIVO LO GUARDO EN UNA TABLA JUNTO A SU NFA
                                cant_Segments_nuevos++;

                                vector<float> Segment;
                                Segment.push_back( (coordenadas1[ c ].x) );
                                Segment.push_back( (coordenadas1[ c ].y) );
                                Segment.push_back( (coordenadas1[ f ].x)  );
                                Segment.push_back( (coordenadas1[ f ].y)  );
                                Segment.push_back( NFA_nueva );

                                Detecciones_EXC.push_back(Segment); // agrego nuevo Segment
                                ExcludeSegment.push_back(false);
                            }

                        }

                        cant_Segments = Detecciones_EXC.size(); //actualizo tamano array
                        if (FLAG_intersect ==1)
                        {
                            FLAG_intersect =0;
                            break;
                        }


                    }//termina if hay interseccion

                }

            } // termino de recorrer todos los Segments 2 con NFA mas chica que i

            i++;

 } // termina loop ppo exclusion

    

        //vector<vector<float> > Detecciones_EXC2; //guardara los Segments ppo exclusion.

        for (size_t ii=0;ii<Detecciones_EXC.size();ii++)
        {
            //if (Detecciones_EXC[ii].empty())
            //  break; //Corrige bug.
            if (ExcludeSegment[ii] != true)
            {

                vector<float> Segment;
                Segment.push_back( Detecciones_EXC[ii][0] ); //X1
                Segment.push_back( Detecciones_EXC[ii][1] ); //Y1
                Segment.push_back( Detecciones_EXC[ii][2] ); //X2
                Segment.push_back( Detecciones_EXC[ii][3] ); //Y2
                Segment.push_back( Detecciones_EXC[ii][4] ); //NFA

                Detecciones_EXCOUT.push_back(Segment); //Ingreso Segment

            }

        }
        /*
        Detecciones_EXC.clear();
        for (size_t ii=0;ii<Detecciones_EXC2.size();ii++)
        {

                Detecciones_EXC.push_back(Detecciones_EXC2[ii]); //Ingreso Segment



          }
    */

        //Detecciones_EXCOUT = Detecciones_EXC2;
        //foo.swap(bar);

        //Detecciones_EXC.swap(Detecciones_EXC2);
    

} //ExlusionPrinciple

void Maximality(std::vector<std::vector<float> > &Detecciones, std::vector<std::vector<float> > &Detecciones_MAX)
{
    vector<bool> VerifyMaximality( Detecciones.size(), true);

    sort(Detecciones.begin(), Detecciones.end(),sortcol2); //Ordena detecciones por su NFA

    int ppo_i, ppo_j, fin_i, fin_j;

    //PRIMER PASADA BORRA SEGMENTOS POCO SIGNIFICATIVOS INCLUIDOS EN SEGMENTOS MAS SINGIFICATIVOS
    for (size_t i=0; i<Detecciones.size()-1;i++){
        if ( VerifyMaximality[i]==true ){
            ppo_i = Detecciones[i][5];
            fin_i = Detecciones[i][6];
            for (size_t j=1+i; j<Detecciones.size();j++){
                if ( VerifyMaximality[j]==true ){
                    ppo_j = Detecciones[j][5];
                    fin_j = Detecciones[j][6];
                    if ( ( ppo_i <= ppo_j) && ( fin_i >= fin_j) )
                    {//J incluido en I y tiene mayor NFA, no lo considero, lo borro
                        VerifyMaximality[j]=false;
                    }
                }//end if j
            } //end FOR j
        } //end IF i
    }// finish FOR i, all the segments analyzed
    //SEGUNDA PASADA BORRA SEGMENTOS POCO SIGNIFICATIVOS QUE INCLUYEN SEGMENTOS QUE SON MAS SINGIFICATIVOS
        for (size_t i=0; i<Detecciones.size()-1;i++){
            if ( VerifyMaximality[i]==true ){
                ppo_i = Detecciones[i][5];
                fin_i = Detecciones[i][6];
                for (size_t j=1+i; j<Detecciones.size();j++){
                    if ( VerifyMaximality[j]==true ){
                        ppo_j = Detecciones[j][5];
                        fin_j = Detecciones[j][6];
                        if ( ( ppo_i >= ppo_j) && ( fin_i <= fin_j) )
                        {//I incluido en J y J tiene mayor NFA, debo borrar j
                            VerifyMaximality[j]=false;
                        }
                    }//end if j
                } //end FOR j
            } //end IF i
        }// finish FOR i, all the segments analyzed

        for (size_t i=0;i<Detecciones.size();i++)
        {
            if (VerifyMaximality[i]==true)
            {
                vector<float> Segment;
                Segment.push_back( Detecciones[i][0] ); //X1
                Segment.push_back( Detecciones[i][1] ); //Y1
                Segment.push_back( Detecciones[i][2] ); //X2
                Segment.push_back( Detecciones[i][3] ); //Y2
                Segment.push_back( Detecciones[i][4] ); //NFA

                Detecciones_MAX.push_back(Segment); //Ingreso Segment Maximal
            }

        }

} //Maximality

void MaximalMeaningfulScratchGrouping(vector<vector<float> > &Detecciones_MAX, const cv::Mat bin, const cv::Mat PM, const int nfaThreshold, const std::vector<std::vector<float> > lines_Hough, const long long int  Ntests, int largo_min)
{

    double EPS=pow (10.0, nfaThreshold);

    for (size_t i=0;i<lines_Hough.size();i++)
    {

        vector<int> perfil; //Segment 1 ----
        vector<Point> coordenadas;
        SegmentIterator(lines_Hough, i, bin, coordenadas, perfil);

        vector<int> extremos(perfil.size());
        vector<int> comienzos(perfil.size());
        vector<int> finales(perfil.size());

        int ind_c=0; int ind_f=0; //iteran sobre cantidad de comienzos y finales respectivos
        if  (perfil[0]==1)
        {//estrategia de borde para ppo array
            comienzos[ind_c]=0;
            ind_c++;
        }

        for ( size_t j =0; j < perfil.size()-1; j++)
        {
            extremos[j]= (int)perfil[j]- (int)perfil[j+1];

            if (extremos[j]== -1)
            { //Detecta un comienzos
                comienzos[ind_c]=j+1; // aparece defasado 1 indice
                ind_c++;
            }

            if (extremos[j]== 1)
            { //Detecta un Final de Segment
                finales[ind_f]=j; // aparece defasado 1 indice
                ind_f++;
            }

        }
        if (perfil[perfil.size()-1]==1)
        { //estrategia de borde para final array
            finales[ind_f]=perfil.size()-1;
            ind_f++;
        }


        //---------PARA CADA Segment AVERIGUO SI ES SIGNIFICATIVO y lo guardo junto a su NFA
        int largo;
        float NFAi;

        vector<vector<float> > Detecciones; //guarda las detecc significativas

        for (int c=0; c<ind_c; c++)
        { //recorro todos los comienzos
            for (int f=c; f<ind_f; f++)
            { //recorro todos los finales

                largo = finales[f]-comienzos[c]+1;

                if ((largo > largo_min)){

                    NFAi= NFA(perfil,coordenadas,comienzos[c],finales[f],PM,Ntests); 

                    if (( NFAi<EPS) && (NFAi!=0))
                    { //Segment SIGNIFICATIVO LO GUARDO EN UNA TABLA JUNTO A SU NFA
                    //cout << "SIGNIFICATICO! Hoeffding = " << H << endl;

                        vector<float> Segment;
                        Segment.push_back( (coordenadas[ comienzos[c] ].x) );
                        Segment.push_back( (coordenadas[ comienzos[c] ].y) );
                        Segment.push_back( (coordenadas[ finales[f] ].x)  );
                        Segment.push_back( (coordenadas[ finales[f] ].y)  );
                        Segment.push_back( NFAi );
                        Segment.push_back( comienzos[c]  );
                        Segment.push_back( finales[f]  );

                        Detecciones.push_back(Segment);

                    }

                }

            }
        }

        //----------------PPO MAXIMALIDAD -------------------------------
        if (Detecciones.size() != 0)
            Maximality(Detecciones, Detecciones_MAX);

    } // termina for todas las lineas de Hough

}


void RemoveScratches(const cv::Mat src, cv::Mat &dst,const int nfaThreshold, const int thresholdHough, const int scratchWidth, const int medianDiffThreshold, const int inclination, const int minLength, const int minDistance,const int linesThickness,const int inpaintingRadius, InpaintingEnum inpaintingMethod, OutputEnum output)
{
//src matriz en RG

    int nRows = src.rows;
    int nCols = src.cols;
    
    Mat src_bw, bin;
    cvtColor(src, src_bw, CV_BGR2GRAY);
    BinaryDetection(src_bw,bin, scratchWidth, medianDiffThreshold); //Deteccion binaria per-pixel

    Mat PM;
    PixelDensity2(bin,PM); //Calculo mapa densidad pixeles 
    //cout << "corre PixelDensity2" << endl;

    //float minLength = round(nRows/10); //largo minimo aceptado para un scratch
    long long int Ntests = (long long int)nRows*nRows*nCols*inclination*4; //Numero de tests para metodologia a contrario
        
    //----------------TRANSFORMADA DE HOUGH -------------------------------

    vector<vector<float> > lines_Hough;
    HoughSpeedUp(bin, thresholdHough, inclination, lines_Hough); //devuelve lineas casi verticales de acuerdo a parametros
    
    vector<vector<float> > Detecciones_MAX; //guardara los Segments signficativos maximales metodologia a contrario.
    MaximalMeaningfulScratchGrouping(Detecciones_MAX,bin,PM,nfaThreshold,lines_Hough,Ntests,minLength);
    /// PRINCIPIO DE EXCLUSION  ---------------------------------------
    //vector<vector<float> > Detecciones_EXC = Detecciones_MAX; //guardara los Segments ppo exclusion
    //ExclusionPrinciple(Detecciones_MAX, Detecciones_EXC,bin,PM,nfaThreshold,Ntests,minLength, minDistance);
   
    vector<vector<float> > Detecciones_EXC = Detecciones_MAX; //guardara los Segments ppo exclusion
    vector<vector<float> > Detecciones_EXCOUT;
    ExclusionPrinciple(Detecciones_EXC,Detecciones_EXCOUT, bin,PM,nfaThreshold, Ntests,minLength, minDistance);
   
    Detecciones_EXC=Detecciones_EXCOUT;


    switch(output)
    {   
        case eDetectionMask: 
        {  
            //----------------IMPRIMO DETECCIONES luego de ppo exclusion -------------------------------
            //Mat cdst3;
            //cvtColor(src, cdst3, CV_GRAY2BGR); //dst2 guarda detecciones maximales
            Mat dst_Exc = Mat::zeros(nRows,nCols, CV_8UC3);
            for (size_t i=0;i<Detecciones_EXC.size();i++){

                int x1=Detecciones_EXC[i][0];
                int y1=Detecciones_EXC[i][1];
                int x2=Detecciones_EXC[i][2];
                int y2=Detecciones_EXC[i][3];

                line( dst_Exc, Point(x1,y1), Point(x2, y2), Scalar(255,255,255), linesThickness, CV_AA);
            }


            dst=dst_Exc;
            
            dst_Exc.release();
            bin.release();                                    
            PM.release();                                    
            src_bw.release();
            break; 

        }// fin de eOverlayDetection
        case eOverlayDetection: 
        {    
							
            //----------------IMPRIMO DETECCIONES luego de ppo exclusion -------------------------------
            //Mat cdst3;
            //cvtColor(src, cdst3, CV_GRAY2BGR); //dst2 guarda detecciones maximales
            
            Mat dst_Exc = src.clone();
            for (size_t i=0;i<Detecciones_EXC.size();i++){

                int x1=Detecciones_EXC[i][0];
                int y1=Detecciones_EXC[i][1];
                int x2=Detecciones_EXC[i][2];
                int y2=Detecciones_EXC[i][3];

                line( dst_Exc, Point(x1,y1), Point(x2, y2), Scalar(0,0,255), linesThickness, CV_AA);
            }
            dst=dst_Exc;
            
            /*
            Mat dst_Exc = src.clone();
            for (size_t i=0;i<Detecciones_MAX.size();i++){

                int x1=Detecciones_MAX[i][0];
                int y1=Detecciones_MAX[i][1];
                int x2=Detecciones_MAX[i][2];
                int y2=Detecciones_MAX[i][3];

                line( dst_Exc, Point(x1,y1), Point(x2, y2), Scalar(0,0,255), linesThickness, CV_AA);
            }

        	dst=dst_Exc;
        	*/
        	
        	//Mat binRGB;
            //cvtColor(bin, binRGB, CV_GRAY2BGR); //dst2 guarda detecciones maximales
        	//dst=binRGB;
            
            dst_Exc.release();
            bin.release();                                    
            PM.release();                                    
            src_bw.release();
        
        break; 

        }// fin de eDetectionMap

        case eRestoration: 
        { 
         
            //----------------IMPRIMO DETECCIONES luego de ppo exclusion -------------------------------
            //Mat cdst3;
            //cvtColor(src, cdst3, CV_GRAY2BGR); //dst2 guarda detecciones maximales
            Mat dst_Exc = Mat::zeros(nRows,nCols, CV_8UC3);
            for (size_t i=0;i<Detecciones_EXC.size();i++){

                int x1=Detecciones_EXC[i][0];
                int y1=Detecciones_EXC[i][1];
                int x2=Detecciones_EXC[i][2];
                int y2=Detecciones_EXC[i][3];

                line( dst_Exc, Point(x1,y1), Point(x2, y2), Scalar(0,0,255), linesThickness, CV_AA);
            }

            //----------------------INPAINTING RESTAURACION--------------------------------------------------------------------
            Mat dst_Inpaint, mask;

            cvtColor(dst_Exc,mask,CV_RGB2GRAY);

            if (inpaintingMethod == eNavier) {
            	inpaint(src,mask,dst_Inpaint,inpaintingRadius,CV_INPAINT_NS);
            }  else if (inpaintingMethod == eFast) {
            	inpaint(src,mask,dst_Inpaint,inpaintingRadius,CV_INPAINT_TELEA);

            }

            dst=dst_Inpaint;
            
            dst_Inpaint.release();
            mask.release();
            dst_Exc.release();
            bin.release();            
            PM.release();             
            src_bw.release();

            break;
        }



    }
        //fin de switch

}








