#define NX %(nx)d
#define NA %(na)d


#define A(i,j) A[i*4 + j]

__kernel
void Solve(__global float4* A, __global float4* X, __global float4* I, float alpha)
{  
    unsigned int ix = get_global_id(0);
    unsigned int ia = get_global_id(1);

    if (I[ix].w < 1e-4) {
    
        float3 c = X[ix].xyz - A(ia,3).xyz;
        float4 a;

        unsigned int i;
        uchar j = 0;
        float q, qsum, I1;
        bool opt;

        opt = true;
        qsum = 1.;
        I1 = 0.;
        
        while ((j < 3) && opt) {
            a = A(ia,j);
            q = a.x*c.x+a.y*c.y+a.z*c.z;
            opt = opt && (q >= 0) && (q <= 1);
            
            if (opt) {
                qsum -= q;
                I1 += a.w*q;
                j++;
            }
 
        }
        opt = opt && (qsum >= 0) && (qsum <= 1);  
        if (opt && (j == 3)) {
            I1 += qsum*A(ia,3).w;
            I[ix].w = alpha*I1;
        }
        
    }       
    
}
             
    
        

                     
                 
                 
    
