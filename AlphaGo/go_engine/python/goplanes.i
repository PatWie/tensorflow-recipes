%module goplanes

%{
    #define SWIG_FILE_WITH_INIT
    #include "goplanes.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (char *STRING, int LENGTH) {(char *str, int strlen)}
%apply (char *STRING, int LENGTH) {(char* bytes, int byteslen)}
%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {(int* data, int dc, int dh, int dw)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* bblack, int bm, int bn)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* bwhite, int wm, int wn)}
%include "goplanes.h"