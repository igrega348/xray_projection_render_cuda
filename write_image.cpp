
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(){
  std::cout << "Hello world" << std::endl;
  char name[] = "image.png";
  std::cout << "File name: " << name << std::endl;
  int h = 256;
  int w = 256;
  std::cout << "h*h/16: " << h*h/16 << std::endl;
  uint8_t data[h*w] = {};
  for (int i = 0; i<h*w; i++){
    data[i] = 250;
  }
  for (int i = 0; i<h; i++){
    for (int j = 0; j<w; j++){
      int dx = j - h/2;
      int dy = i - w/2;
      int r2 = dx*dx + dy*dy;
      float decision = (float) sqrt(r2 - h*h/16);
      uint8_t v = (uint8_t) (255.0f / (1.0 + exp(-decision/10.0)));
      data[i*w + j] = v;
      // if (r2<=h*h/16){
      //   data[i*w + j] = 0;
      // }
    }
  }
  std::cout << "Stride: " << sizeof(data[0]) << std::endl;
  std::cout << "Array: " << sizeof(data) << std::endl;
  int stride = w;
  stbi_write_png_compression_level = 1;
  stbi_write_png(name, w, h, stbi_write_png_compression_level, &data, stride);
}

