#include <iostream>
#include <bits/stdc++.h>

int main(int argc, char* argv[])
{
    if (argc < 2) { 
        std::cerr<<"usage: "<<argv[0]<<" <n> <dt>"<<std::endl;
        return -1;
    }
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int n = atoi(argv[1]);
    float dt = atof(argv[2]);

    float t_max = 2.0;
    float x_min = 0.0;
    float x_max = 1.0;
    float v = 1;
    float x_c = 0.25;

    float dx = (x_max - x_min) / 2;
    float nb_steps = t_max/dt;
    float alpha = (v*dt)/(2*dx);

    float x[n+2];
    float u_0[n+2];
    float u[n+2];
    float u_new[n+2];
    //xmin + (i − 1) ∗ dx, ∀i ∈ {0 . . . N + 2};
    for(int i=0; i<=n+2; i++){
        x[i] = x_min + (i-1) * dx;
    }
    //u0[i] = e−200∗(x[i]−xc)2, ∀i ∈ {0 . . . N + 2};
    for(int i=0; i<=n+2; i++){
        u_0[i] = std::exp(-200*(x[i] - x_c)*(x[i] - x_c));
    }
    memcpy(u, u_0, (n+2)*sizeof(float));
    memcpy(u_new, u_0, (n+2)*sizeof(float)); 

    int current_time;

    for (int timestamp=0; timestamp<nb_steps; timestamp++){
        current_time = timestamp*dt;
        for (int j=1; j<n+2; j++){
            u_new[j] = u[j] - alpha*(u[j+1] - u[j-1]) + 0.5*(u[j+1] - 2*u[j] + u[j-1]);
        }
        memcpy(u, u_new, (n+2)*sizeof(float));

        u[0] = u[n+1];
        u[n+2] = u[1];
        // std::cout << u[0] << " " << u[n+2] << std::endl;
        // if (timestamp%1000 == 0){
        //     std::cout << timestamp << " " << nb_steps << "\n";
        // }
    }

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elpased_seconds = end-start;
    std::cerr<<"Time taken for n=" << n << " & dt="<< dt << " : " << elpased_seconds.count()<<std::endl;
}


