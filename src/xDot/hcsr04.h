/* Copyright (c) 2013 Prabhu Desai
 * pdtechworld@gmail.com
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * For more details on the sensor :
 * http://www.elecfreaks.com/store/hcsr04-ultrasonic-sensor-distance-measuring-module-p-91.html?zenid=pgm8pgnvaodbe36dibq5s1soi3
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef MBED_HCSR04_H
#define MBED_HCSR04_H

#include "mbed.h"

/** HCSR04 Class(es)
 */

class HCSR04
{
public:
    /** Create a HCSR04 object connected to the specified pin
    * @param pin i/o pin to connect to
    */
    HCSR04(PinName TrigPin,PinName EchoPin);
    ~HCSR04();

    /** Return the distance from obstacle in cm
    * @param distance in cms and returns -1, in case of failure
    */
    unsigned int get_dist_cm(void);
    /** Return the pulse duration equal to sonic waves travelling to obstacle and back to receiver.
    * @param pulse duration in microseconds.
    */
    unsigned int get_pulse_us(void);
    /** Generates the trigger pulse of 10us on the trigger PIN.
    */
    void start(void );
    void isr_rise(void);
    void isr_fall(void);
    void fall (void (*fptr)(void));
    void rise (void (*fptr)(void));



private:

    Timer pulsetime;
    DigitalOut  trigger;
    InterruptIn echo;
    unsigned int pulsedur;
    unsigned int distance;
};

#endif