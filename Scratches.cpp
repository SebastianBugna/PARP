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


//----------------FUNCION AUXILIAR CALCULAR NFA -------------------------------
float NFA(const vector<int> perfil, const vector<cv::Point> coordenadas, const int c, const int f, const cv::Mat  PM , const long long int  Ntests )
{

	int largo= f-c+1;;
	double pm =0; //densidad promedio de todos los pixels de la recta
	double k0,r; //parametros cota de Hoeffding
	r=k0=0;
	float NFA =0; //Cota de Hoeffding.


	//vector que guarda extremos y NFA de cada segmento detectado

	for (int ind = c; ind<=f;ind++)
	{ //hallo densidad promedio de segmento

		k0+=perfil[ind];
		pm+= PM.at<double>(coordenadas[ ind ]);
	}
	pm=pm/largo;
	r=k0/largo;

	NFA=  exp(  -(largo)* ( (r*log(r/pm)) + (   (1-r)*log((1-r)/(1-pm))) )   );
	if (r==1)
	{
		NFA=exp(  -(largo)*(r*log(r/pm))   ); // hallo cota de hoeffding, si es menor a cero el segmento es significativo
	}

	NFA=NFA*Ntests; //NFA

	return NFA;
}

//----------------BINARIZACION DE IMAGEN -------------------------------
void BinaryDetection(cv::Mat src, cv::Mat &dst){
//int channels = src.channels();
        int nRows = src.rows;
        int nCols = src.cols;

        Mat bin = Mat::zeros(nRows,nCols, CV_8U);
        Mat Ig;
        GaussianBlur(src, Ig, Size(3,3), 1.0, 1.0,0  );
        //src = Ig.clone();

        int s_med=3;
        int s_avg=20;
        int tmpPix_g;
        int tmpPix_m;
        float Avg_L;
        float Avg_R;

        for(int y = 0; y < nRows; y++){ //recorro imagen Src
            for ( int x = 0; x < nCols; x++){

                //cout << "srcPix " << (int)src.at<uchar>(y,x) << endl;

                if ((x > 5) && (x < nCols - 5)  && (y > 1) && (y < nRows- 1)) {

                    tmpPix_g= (int)Ig.at<uchar>(y,x);

                    int srcPix5 = (int)Ig.at<uchar>(y,x-1);
                    int srcPix6 = (int)Ig.at<uchar>(y,x+1);

                    //Mediana horizontal hardcodeada!! Arreglar
                    int srcPix10 = (int)Ig.at<uchar>(y,x+2);
                    int srcPix11 = (int)Ig.at<uchar>(y,x-2);

                    int med_h[4] = {srcPix5,srcPix6,srcPix10,srcPix11};
                    std::sort(med_h, med_h+4);

                    tmpPix_m = med_h[1];  //<------------------------ poner un 1 de nuevo-----------------------------------------

                    //cout << "tmpPix_m " << tmpPix_m << endl;

                    //Promedios laterales

                    int srcPix_L1 = (int)Ig.at<uchar>(y,x-3);
                    int srcPix_L2 = (int)Ig.at<uchar>(y,x-4);
                    int srcPix_L3 = (int)Ig.at<uchar>(y,x-5);
                    int srcPix_R1 = (int)Ig.at<uchar>(y,x+3);
                    int srcPix_R2 = (int)Ig.at<uchar>(y,x+4);
                    int srcPix_R3 = (int)Ig.at<uchar>(y,x+5);


                    Avg_L = (float)(srcPix_L1+srcPix_L2+srcPix_L3)/(float)3;
                    Avg_R = (float)(srcPix_R1+srcPix_R2+srcPix_R3)/(float)3;

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

            }
        }//Fin de Binarizacion
        //imshow("Binarizacion",bin);

        dst = bin;

}

void PixelDensity(cv::Mat bin,cv::Mat &PM){

    int nRows = bin.rows;
    int nCols = bin.cols;
    PM = Mat::zeros(nRows,nCols, CV_64FC1);

    

        //----------------MAPA DENSIDAD PIXELES IMAGEN -------------------------------
        //algunas constantes que utiliza el algoritmo
        float largo_min = round(nRows/10); //largo minimo aceptado para un scratch

        long long int Ntests = (long long int)nRows*nRows*nCols*40; //Numero de tests para metodo a contrario
        //cout<< "pixels = "<< Ntests << endl;


        
        //CV_8UC1

        int L = (int)(nCols/30);
        float max_density=0;
        float density;
        int lim_y, lim_x, cant_pixels;

        for(int i = 0; i < nRows; ++i){
            for ( int j = 0; j < nCols; ++j){
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

void RestoreScratches(cv::Mat src, cv::Mat &dst, bool detectionMap,  bool original,  bool restored, int thresholdHough, int inclination, int inpaintingRadius, int inpaintingMethod){

        

        int nRows = src.rows;
        int nCols = src.cols;
        
        //EMPIEZA PROCESAMIENTO DE DETECTION
        //------------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------------
        //------------------------------------------------------------------------------------------


        BinaryDetection(src,dst);
        


        Mat cdst, PM;

        PixelDensity(dst,PM);

        float largo_min = round(nRows/10); //largo minimo aceptado para un scratch

        long long int Ntests = (long long int)nRows*nRows*nCols*40; //Numero de tests para metodo a contrario
        


        //----------------TRANSFORMADA DE HOUGH -------------------------------
          cvtColor(src, cdst, CV_GRAY2BGR); //la imagen de salida cdst es en colores

          vector<vector<float> > Detecciones_MAX; //guardara los segmentos signficativos maximales metodologia a contrario.

          vector<Vec2f> lines;
          //HoughLines(InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn=0, double stn=0 )
          HoughLines(dst, lines, 1, CV_PI/180, thresholdHough);

          //float largo_seg;

          for( size_t i = 0; i < lines.size(); i++ ) //ITERO PARA CADA LINEA
          {

        	  float rho = lines[i][0], theta = lines[i][1];
        	  Point pt1, pt2;
        	  double a = cos(theta), b = sin(theta);
        	  double x0 = a*rho, y0 = b*rho;
        	  pt1.x = cvRound(x0 + 1000*(-b));
        	  pt1.y = cvRound(y0 + 1000*(a)); //hardcodign cambiar el 1000 no funciona para HD por ejemplo <<<< <----------------
        	  pt2.x = cvRound(x0 - 1000*(-b));
        	  pt2.y = cvRound(y0 - 1000*(a));

        	  //largo_seg = sqrt( pow((pt1.x-pt2.x),2) + pow(( pt1.y-pt2.y),2));

            if ((fabs(theta* 180 / CV_PI)<=1)  ) // caduco && (largo_seg > largo_min) )   // solo acepto rectas con inclinacion menor a 10 grados
            {

                //dibujo rectas detectadas
            	line( cdst, pt1, pt2, Scalar(0,0,255), 2, CV_AA);
                //line( cdst, Point(l[0],l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, CV_AA);


            	//cout << "PPO: " << pt1.x << "," << pt1.y << " FIN: " <<  pt2.x << "," << pt2.y << endl;


                //---------------- DETECCION SEG SIGNIFICATIVOS -------------------------------

                //LLAMO A LINEITERATOR // grabs pixels along the line (pt1, pt2)
                // from 8-bit unsigned char image (binary) to the buffer.
                LineIterator it(dst, pt1, pt2, 4);
                LineIterator it2 = it;

                int perfil[it.count]; //creo buffer para guardar pixeles binarios.
                vector<Point> coordenadas(it.count);

                // iterating through the line
                for(int i = 0; i < it2.count; i++, ++it2)
                {
                   //---------OBTENGO PERFIL DE RECTA Y SUS COORD DE PIXEL EN LA IMAGEN
                   perfil[i] =  (int)dst.at<uchar>( it2.pos() )/255; //Sustituir por MaxValue Natron
                   coordenadas[i] = it2.pos(); // guarda coord pixel en la imgen
                   //cout << "salida= " << (int)perfil[i] << endl;
                   //cout <<perfil[i] ; //imprime array con flag de extremos
                }
                //---------HALLO COMIENZO Y FIN DE SEGMENTOS DENTRO DEL PERFIL
                int extremos[it.count];
                int comienzos[it.count];
                int finales[it.count];
                int ind_c=0; int ind_f=0; //iteran sobre cantidad de comienzos y finales respectivos
                //cout << endl <<"extremos del perfil anterior"<< endl;
                if  (perfil[0]==1)
                {//estrategia de borde para ppo array
					//cout << endl <<"FLAG COMIENZO =1 " << endl ;
					comienzos[ind_c]=0;
					ind_c++;
                }

                for ( int j =0; j < it2.count-1; j++){
                    extremos[j]= (int)perfil[j]- (int)perfil[j+1];

                    //cout <<(int)extremos[j] ; //imprime array con flag de extremos

                    if (extremos[j]== -1)
                    { //Detecta un Comienzo
                    	comienzos[ind_c]=j+1; // aparece defasado 1 indice
                    	ind_c++;
                    }

                    if (extremos[j]== 1)
                    { //Detecta un Final de segmento
                       finales[ind_f]=j; // aparece defasado 1 indice
                       ind_f++;
                    }

                }
                if (perfil[it.count-1]==1)
                { //estrategia de borde para final array

                    finales[ind_f]=it.count-1;
                    ind_f++;
                }


                //---------PARA CADA SEGMENTO AVERIGUO SI ES SIGNIFICATIVO y lo guardo junto a su NFA
                int largo;
                double pm; //densidad promedio de todos los pixels de la recta
                double k0,r; //parametros cota de Hoeffding
                float H; //Cota de Hoeffding.

                vector<vector<float> > Detecciones; //guarda las detecc significativas

                for (int c=0; c<ind_c; c++){ //recorro todos los comienzos
                	for (int f=c; f<ind_f; f++){ //recorro todos los finales

                		largo = finales[f]-comienzos[c]+1;

                		//cout << " SEGMENTO ANALIZADO: " << endl << " Comienza en el indice de PERFIL = " <<  comienzos[c] << endl ;
                		//cout << " Coords de comienzo = " << coordenadas[ comienzos[c] ] << endl << " Coords de final= " << coordenadas[ finales[f] ] << endl ;
                		//cout << " Termina en el indice de PERFIL = " << finales[f] << endl << " Largo = " << largo << endl ;

                		pm=0;
                		r=k0=0;
                		H=0;

                		//vector que guarda extremos y NFA de cada segmento detectado

                		for (int ind = comienzos[c]; ind<finales[f];ind++)
                		{ //hallo densidad promedio de segmento
                			//cout  << " Pm.atcoords " << PM.at<double>(coordenadas[ ind ]) << endl ;
                			k0+=perfil[ind];
                			pm+= PM.at<double>(coordenadas[ ind ]);

                		}
                		pm=pm/largo;
                		//cout << pm << endl;
                		r=k0/largo;


                		if (r==1)
                			H=exp(  -(largo)*(r*log(r/pm))   ); // hallo cota de hoeffding, si es menor a cero el segmento es significativo
                		else
                			H=  exp(  -(largo)* ( (r*log(r/pm)) + (   (1-r)*log((1-r)/(1-pm))) )   );

                		H=H*Ntests; //NFA

                		//cout << endl << "DENSIDAD PROMEDIO = " << pm << endl;
                		//cout << "r = " << r << endl;
                		//cout << "Hoeffding = " << (float)H << endl;

                		if (( H<1) && (H!=0) && (largo > largo_min))
                		{ //SEGMENTO SIGNIFICATIVO LO GUARDO EN UNA TABLA JUNTO A SU NFA
                			//cout << "SIGNIFICATICO! Hoeffding = " << H << endl;

                			vector<float> Segmento;
                			Segmento.push_back( (coordenadas[ comienzos[c] ].x) );
                			Segmento.push_back( (coordenadas[ comienzos[c] ].y) );
                			Segmento.push_back( (coordenadas[ finales[f] ].x)  );
                			Segmento.push_back( (coordenadas[ finales[f] ].y)  );
                			Segmento.push_back( H );
                			Segmento.push_back( comienzos[c]  );
                			Segmento.push_back( finales[f]  );

                			Detecciones.push_back(Segmento);

                		}

                	}
                }

                //cout << "tamano vector significativos:" << Detecciones.size() << endl << "tamano de cada vector:" << Detecciones[0].size() << endl	;
                sort(Detecciones.begin(), Detecciones.end(),sortcol); //Ordena detecciones por su NFA
                /*
                cout << "ORDENO ---> " << endl;
                for (int ii = 0; ii < Detecciones.size(); ii++){ //Imprime detecciones
                	cout << "------------------- " << endl;
                  for (int jj = 0; jj < Detecciones[ii].size(); jj++){
                     cout << Detecciones[ii][jj] << endl;
                }
                  }
				*/

                //----------------PPO MAXIMALIDAD -------------------------------

                for (int ii=Detecciones.size()-1; ii>0;--ii){


                	if (Detecciones.size() == 0)
                		break;

                	if ( Detecciones[ii][4] != 999){ //se "borra" segmento si NFA = 999

                		float NFA_I=Detecciones[ii][4];
                		int ppo_I = Detecciones[ii][5];
                		int fin_I = Detecciones[ii][6];

                		//for ( size_t jj=4;jj>0;jj--){
                		for ( int jj=Detecciones.size()-1-ii;jj>0;--jj){

        					if (( ii != jj)  && ( Detecciones[jj][4] != 999)){ //entonces comparo ambos

        						float NFA_J=Detecciones[jj][4];
        						int ppo_J = Detecciones[jj][5];
        						int fin_J = Detecciones[jj][6];



        						if ( (NFA_J>NFA_I) && ( ppo_I <= ppo_J) && ( fin_I >= fin_J) ){
        						//J incluido en I y tiene mayor NFA, no lo considero, lo borro
        							Detecciones[jj][4]=999;
        						}
        						else if ( (NFA_J>NFA_I) && ( ppo_I >= ppo_J) && ( fin_I <= fin_J) ){
        						//I incluido en J y J tiene mayor NFA, debo borrar j
        							Detecciones[jj][4]=999;
        						}




        						/*
        						if ( (NFA_I<NFA_J) && ( ppo_I >= ppo_J) && ( fin_I <= fin_J) ){
        						//I incluido en J y tiene menor NFA, debo borrar j
        							Detecciones[jj][4]=999;
        						}

        						if ( (NFA_J>NFA_I) && ( ppo_I <= ppo_J) && ( fin_I >= fin_J) ){
        						//J incluido en I y tiene mayor NFA, no lo considero, lo borro
        						Detecciones[jj][4]=999;
        						}*/

        					}
                		 }
                	}
                }//termino de borrar segmentos que no son maximales

                for (size_t ii=0;ii<Detecciones.size();ii++){

                	if (Detecciones[ii][4]!=999){

                		vector<float> Segmento;
                		Segmento.push_back( Detecciones[ii][0] ); //X1
                		Segmento.push_back( Detecciones[ii][1] ); //Y1
                		Segmento.push_back( Detecciones[ii][2] ); //X2
                		Segmento.push_back( Detecciones[ii][3] ); //Y2
                		Segmento.push_back( Detecciones[ii][4] ); //NFA

                		Detecciones_MAX.push_back(Segmento); //Ingreso Segmento Maximal

                	}

                }

            } //termina if theta < 10 grados
          } // termina for todas las lineas de Hough

          /*
          cout << "MAXIMALES ---> " << endl;
          for (size_t ii = 0; ii < Detecciones_MAX.size(); ii++){ //Imprime detecciones
          	cout << "------------------- " << endl;
            for (size_t jj = 0; jj < Detecciones_MAX[ii].size(); jj++){
               cout << Detecciones_MAX[ii][jj] << endl;
          }
            }
          */

        //----------------IMPRIMO DETECCIONES MAXIMALES -------------------------------
        Mat cdst2;
        cvtColor(src, cdst2, CV_GRAY2BGR); //dst2 guarda detecciones maximales
         //Mat cdst2 = Mat::zeros(nRows,nCols, CV_8UC3);
        for (size_t i=0;i<Detecciones_MAX.size();i++){

        	int x1=Detecciones_MAX[i][0];
        	int y1=Detecciones_MAX[i][1];
        	int x2=Detecciones_MAX[i][2];
        	int y2=Detecciones_MAX[i][3];

        	line( cdst2, Point(x1,y1), Point(x2, y2), Scalar(255,255,255), 1, CV_AA);
        }

        //ExclusionPrinciple(std::vector<std::vector<float>> Detecciones_MAX, );

        /// PRINCIPIO DE EXCLUSION  ---------------------------------------

        vector<vector<float> > Detecciones_EXC = Detecciones_MAX; //guardara los segmentos signficativos maximales metodologia a contrario.

        sort(Detecciones_EXC.begin(), Detecciones_EXC.end(),sortcol2); //ORDENO NFA DESCENDENTE
 
        	/*
                  cout << "MAXIMALES ---> " << endl;
                  for (size_t ii = 0; ii < Detecciones_EXC.size(); ii++){ //Imprime detecciones
                  	cout << "------------------- " << endl;
                    for (size_t jj = 0; jj < Detecciones_EXC[ii].size(); jj++){
                       cout << Detecciones_EXC[ii][jj]  << endl;
                  }
                    }
			*/


        vector<int> intersecciones;
        bool FLAG_intersect=0;
        //int i=Detecciones_EXC.size()-1;
        //dst es la imagen que contiene la binzarizacion inicial
        //while ( i>0 ){
        //for ( int i = Detecciones_EXC.size()-1;i>0;--i ){
        int cant_segmentos=Detecciones_EXC.size()-1;
        int i=1;

        while ( i<cant_segmentos )
        {

        	if (Detecciones_EXC.size() == 0)
        		break;

        	sort(Detecciones_EXC.begin()+i, Detecciones_EXC.end(),sortcol2);

        	//SEGMENTO 1 ----
        	Point pt1,pt2;
        	pt1.x=Detecciones_EXC[i][0];
           	pt1.y=Detecciones_EXC[i][1];
        	pt2.x=Detecciones_EXC[i][2];
        	pt2.y=Detecciones_EXC[i][3];
        	//float NFA1=Detecciones_EXC[i][4];
            LineIterator it1(dst, pt1, pt2, 8); // LineIterator Segmento 1
            vector<Point> coordenadas1(it1.count);
            vector<int> perfil1(it1.count);; //creo buffer para guardar pixeles binarios.

	  	    for(int jj = 0; jj < it1.count; jj++, ++it1)
	  	    {  //---------OBTENGO PERFIL DE RECTA Y SUS COORD DE PIXEL EN LA IMAGEN
	  	        perfil1[jj] =  (int)dst.at<uchar>( it1.pos() )/255; //Sustituir por MaxValue Natron
	  	        coordenadas1[jj] = it1.pos(); // guarda coord pixel en la imgen
	  	        //cout << "salida= " << (int)perfil[i] << endl;
	  	        //cout <<perfil[i] ; //imprime array con flag de extremos
	  	    }

	  	    for (int j=i-1;j>=0;--j) {

	  	    	if (Detecciones_EXC[j][4] != 999)
	  	    	{

	  	    		//SEGMENTO 2 ----
	  	    		Point pt3,pt4;
	  	    		pt3.x=Detecciones_EXC[j][0];
	  	    		pt3.y=Detecciones_EXC[j][1];
	  	    		pt4.x=Detecciones_EXC[j][2];
	  	    		pt4.y=Detecciones_EXC[j][3];
	  	    		float NFA2=Detecciones_EXC[j][4];
	  	    		LineIterator it2(dst, pt3, pt4, 8); // LineIterator Segmento 1
	  	    		vector<Point> coordenadas2(it2.count);
	  	    		vector<int> perfil2(it2.count);

	  	    		for(int jj = 0; jj < it2.count; jj++, ++it2)
	  	    		{  //---------OBTENGO PERFIL DE RECTA Y SUS COORD DE PIXEL EN LA IMAGEN
	  	    			perfil2[jj] =  (int)dst.at<uchar>( it2.pos() )/255; //Sustituir por MaxValue Natron
	  	    			coordenadas2[jj] = it2.pos(); // guarda coord pixel en la imgen
	  	    		}

	  	    		//RECORRO AMBOS Y BUSCO PIXELES REPETIDOS
	  	    		bool cercanos=0;
	  	    		int distancia =0;
	  	    		int cant_intersecciones=0;
	  	    		for (size_t ii=0;ii<coordenadas1.size();ii++){
	  	    			for (size_t jj=0;jj<coordenadas2.size();jj++){


	  	    				//if (coordenadas1[ii]==coordenadas2[jj])
	  	    				//cout << abs(coordenadas1[ii].x- coordenadas2[jj].x) << endl;
	  	    				//cout << coordenadas2[ii].x << endl;
	  	    				if (( abs(coordenadas1[ii].x- coordenadas2[jj].x)<=10) || (coordenadas1[ii]==coordenadas2[jj]))  /*&&( abs(coordenadas1[ii].y- coordenadas2[jj].y)<=1))*/

	  	    					cercanos=1;

	  	    				if(cercanos==1)
	  	    				{ //<--------------Agregar radio bola Thau_x
	  	    					FLAG_intersect=1;
	  	    					perfil1[ii]=999;
	  	    					intersecciones.push_back(ii);
	  	    					cant_intersecciones++;
	  	    					cercanos=0;
	  	    				}
	  	    			}
	  	    		}

	  	    		if (cant_intersecciones !=0)
	  	    		{


	  	    			int k =1; //recalculo NFA
	  	    			int c,f;
	  	    			c=f=-1;
	  	    			float NFA_nueva;
	  	    			int cant_segmentos_nuevos = 0;

	  	    			if ( perfil1[0]!=999)
	  	    				c=0;

	  	    			while (k<it1.count-1)
	  	    			{
	  	    				if ( ( perfil1[k]!=999) && (perfil1[k-1]==999))
	  	    					c=k;

	  	    				if ( ( perfil1[k]!=999) && (perfil1[k+1]==999))
	  	    					f=k;

	  	    				if ( ( k==it1.count-2) && (perfil1[k+1]!=999)) //BORD
	  	    					f=k+1;

	  	    				if ((c!=-1)&&(f!=-1))
	  	    				{

								// tengo comienzo y tengo final
	  	    					//recalculo NFA
	  	    					/*cout << "c="<< c<< endl;
	  	  					cout << "f="<< f<< endl;
	  	  					cout << "Segmento con indice"<< i-1<< endl;
	  	  					cout << "------"<< (float)NFA_nueva << endl;
	  	    					 */

	  	    					NFA_nueva= NFA(perfil1,coordenadas1,c,f,PM,Ntests);

	  	    					if (( NFA_nueva<1) && (NFA_nueva!=0) && (f-c+1 > largo_min))
	  	    					{//SEGMENTO SIGNIFICATIVO LO GUARDO EN UNA TABLA JUNTO A SU NFA
	  	    						//cout << "------"<< (float)NFA_nueva << endl;
	  	    						//cout << "indice i = "<< i << endl;
	  	    						//cout << "indice j = "<< j << endl;
	  	    						cant_segmentos_nuevos++;
	  	    						Detecciones_EXC[i][4]=999;
	  	    						vector<float> Segmento;
	  	    						Segmento.push_back( (coordenadas1[ c ].x) );
	  	    						Segmento.push_back( (coordenadas1[ c ].y) );
	  	    						Segmento.push_back( (coordenadas1[ f ].x)  );
	  	    						Segmento.push_back( (coordenadas1[ f ].y)  );
	  	    						Segmento.push_back( NFA_nueva );

	  	    						Detecciones_EXC.push_back(Segmento); // agrego nuevo segmento
	  	    					}

	  	    					c=f=-1; //reinicializo comienzo y fin
	  	    				} // termino de agregar segmento

	  	    				k++;

	  	    			}//termino de recorrer segmento 2

	  	    			cant_segmentos = Detecciones_EXC.size(); //actualizo tamano array
	  	    			if (FLAG_intersect ==1)
	  	    			{
	  	    				FLAG_intersect =0;
	  	    				break;
	  	    			}


	  	    		}//termina if hay interseccion

	  	    	}

	  	    } // termino de recorrer todos los segmentos 2 con NFA mas chica que i

	  	    i++;

 } // termina loop ppo exclusion

    

        vector<vector<float> > Detecciones_EXC2; //guardara los segmentos signficativos maximales metodologia a contrario.

        for (size_t ii=0;ii<Detecciones_EXC.size();ii++)
        {

        	if (Detecciones_EXC[ii][4]!=999)
        	{

        		vector<float> Segmento;
        		Segmento.push_back( Detecciones_EXC[ii][0] ); //X1
        		Segmento.push_back( Detecciones_EXC[ii][1] ); //Y1
        		Segmento.push_back( Detecciones_EXC[ii][2] ); //X2
        		Segmento.push_back( Detecciones_EXC[ii][3] ); //Y2
        		Segmento.push_back( Detecciones_EXC[ii][4] ); //NFA

        		Detecciones_EXC2.push_back(Segmento); //Ingreso Segmento Maximal

        	}

        }

        //----------------IMPRIMO DETECCIONES luego de ppo exclusion -------------------------------
               //Mat cdst3;
               //cvtColor(src, cdst3, CV_GRAY2BGR); //dst2 guarda detecciones maximales
                 Mat cdst3 = Mat::zeros(nRows,nCols, CV_8UC3);
               for (size_t i=0;i<Detecciones_EXC2.size();i++){

               	int x1=Detecciones_EXC2[i][0];
               	int y1=Detecciones_EXC2[i][1];
               	int x2=Detecciones_EXC2[i][2];
               	int y2=Detecciones_EXC2[i][3];

               	line( cdst3, Point(x1,y1), Point(x2, y2), Scalar(255,255,255), 2, CV_AA);
               }


  //----------------------INPAINTING RESTAURACION--------------------------------------------------------------------
          Mat dst2, cdst4;
          cvtColor(cdst3,cdst4,CV_RGB2GRAY);

      inpaint(src,cdst4, dst2, 2,CV_INPAINT_TELEA);


dst=dst2;
  //------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------
  /// TERMINA PROCESAR VIDEO---------------------------------------

}
