/*
 * stereoprotocol.h
 *
 *  Created on: Sep 23, 2015
 *      Author: roland
 */

#ifndef SW_AIRBORNE_SUBSYSTEMS_GPS_STEREOPROTOCOL_H_
#define SW_AIRBORNE_SUBSYSTEMS_GPS_STEREOPROTOCOL_H_
#include <stdio.h>
#include <iostream>
#include <inttypes.h>
#include <errno.h>
#include <termios.h>
#include <unistd.h>
struct MsgProperties {
  uint16_t positionImageStart;
  uint8_t width;
  uint8_t height;
} ;
typedef struct MsgProperties MsgProperties;



// function primitives
uint16_t stereoprot_add(uint16_t, uint16_t,uint16_t);
uint16_t stereoprot_diff(uint16_t, uint16_t,uint16_t);
uint8_t stereoprot_isEndOfMsg(uint8_t *, uint16_t,uint16_t);
uint8_t stereoprot_isStartOfMsg(uint8_t *, uint16_t,uint16_t);
void stereoprot_get_msg_properties(uint8_t *, MsgProperties *, uint16_t,uint16_t);
void SendArray(int ,uint8_t* , uint8_t , uint8_t );
#endif /* SW_AIRBORNE_SUBSYSTEMS_GPS_STEREOPROTOCOL_H_ */
