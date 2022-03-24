#include <iostream>
#include <omp.h>   // NEW ADD

using namespace std;


#define DEFINE_idx auto idx = omp_get_thread_num();


int main()
{
        int num = 1;

        // #pragma omp parallel for num_threads(8)
        // #pragma omp parallel  num_threads(8)
        #pragma omp parallel
        {
        //       #pragma omp for reduction(+:num)
                // #pragma omp for schedule(dynamic)
                // #pragma omp for
                        // for(int i=0; i<10; i++) {
                        //         // #pragma omp critical
                        //         // #pragma omp atomic
                        //         num += 1;
                        //         auto idx = omp_get_thread_num();

                        //         // #pragma omp critical
                        //         cout << idx << endl;
                        //         // cout << num << endl;
                        //         // cout << "nihao" << endl;
                        // }



                // #pragma omp for
                //         for(int i=10; i<20; i++) {
                //                 // #pragma omp critical
                //                 // #pragma omp atomic
                //                 num += 1;
                //                 // auto idx = omp_get_thread_num();
                //                 // cout << idx << endl;
                //                 cout << "hello" << endl;
                //         }


                #pragma omp sections
                {
                        cout << "hello world\n" << endl;
                }

                #pragma omp sections
                {
                        cout << "nihao" << endl;
                }
                
                // auto idx = omp_get_thread_num();
                #pragma omp section
                cout << "china" << endl;
        }


        // auto idx = omp_get_thread_num();
        // cout << idx << endl;
        return 0;
}