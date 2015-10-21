/*
 * stereoprotocol.cpp
 *
 *  Created on: Sep 23, 2015
 *      Author: roland
 */

#include "stereoprotocol.h"




/**
 * Increment circular buffer counter by i
 */
uint16_t stereoprot_add(uint16_t counter, uint16_t i, uint16_t buffer_size)
{
  return (counter + i) % buffer_size;
}

/**
 * Decrement circular buffer counter by i
 */
uint16_t stereoprot_diff(uint16_t counter, uint16_t i,uint16_t buffer_size)
{
  return (counter - i + buffer_size) % buffer_size;
}

/**
 * Checks if the sequence in the array is equal to 255-0-0-171,
 * as this means that this is the end of an image
 */
uint8_t stereoprot_isEndOfMsg(uint8_t *stack, uint16_t i,uint16_t buffer_size)
{

  if (stack[i] == 255 && (stack[stereoprot_add(i, 1,buffer_size)] == 0) && (stack[stereoprot_add(i, 2,buffer_size)] == 0) && stack[stereoprot_add(i, 3,buffer_size)] == 171) {
	  return 1;
  }
  return 0;
}

/**
 * Checks if the sequence in the array is equal to 255-0-0-171,
 * as this means a new image is starting from here
 */
uint8_t stereoprot_isStartOfMsg(uint8_t *stack, uint16_t i,uint16_t buffer_size)
{
  if (stack[i] == 255 && (stack[stereoprot_add(i, 1,buffer_size)] == 0) && (stack[stereoprot_add(i, 2,buffer_size)] == 0) && stack[stereoprot_add(i, 3,buffer_size)] == 175) {
    return 1;
  }
  return 0;
}


void WritePart(int fd,uint8_t* code,uint8_t length){

    for(uint8_t index=0; index < length; index++){
        uint8_t toWrite = code[index];
        printf("Writing %i \n",toWrite);
        write(fd,&toWrite,1);
    }

}
void SendArray(int fd,uint8_t* b, uint8_t array_width, uint8_t array_height) {
      uint8_t code[4];
      code[0] = 0xff;
      code[1] = 0x00;
      code[2] = 0x00;
      code[3] = 0xAF; // 175
      WritePart(fd,code,4);


    int horizontalLine = 0;
    for (horizontalLine = 0; horizontalLine < array_height; horizontalLine++) {
        code[3] = 0x80;//128
          WritePart(fd,code,4);
          WritePart(fd,b+array_width*horizontalLine,array_width);

        code[3] = 0xDA;//218
          WritePart(fd,code,4);
    }

    code[3] = 0xAB;
    WritePart(fd,code,4);
}



/**
 * Retrieve size of image from message
 */
void stereoprot_get_msg_properties(uint8_t *raw, MsgProperties *properties, uint16_t start,uint16_t buffer_size)
{
  *properties = (MsgProperties) {start, 0, 0};
  uint16_t i = start, startOfLine = start;
  while (1) {
    // Check the first 3 bytes for the pattern 255-0-0, then check what special byte is encoded next
    if ((raw[i] == 255) && (raw[stereoprot_add(i, 1,buffer_size)] == 0) && (raw[stereoprot_add(i, 2,buffer_size)] == 0)) {
      if (raw[stereoprot_add(i, 3,buffer_size)] == 171) { // End of image
        break;
      }
      if (raw[stereoprot_add(i, 3,buffer_size)] == 128) { // Start of line
        startOfLine = i;
      }
      if (raw[stereoprot_add(i, 3,buffer_size)] == 218) { // End of line
        properties->height++;
        properties->width = stereoprot_diff(i, startOfLine + 4,buffer_size); // removed 4 for the indication bits at the end of line
      }
    }
    i = stereoprot_add(i, 1,buffer_size);
  }
}
