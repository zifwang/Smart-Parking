/* Copyright (c) 2013 Prabhu Desai
 * pdtechworld@gmail.com
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#include "hcsr04.h"


HCSR04::HCSR04(PinName TrigPin,PinName EchoPin):
    trigger(TrigPin), echo(EchoPin)
{
    pulsetime.stop();
    pulsetime.reset();
    echo.rise(this,&HCSR04::isr_rise);
    echo.fall(this,&HCSR04::isr_fall);
    trigger=0;
}

HCSR04::~HCSR04()
{
}

void HCSR04::isr_rise(void)
{
    pulsetime.start();
}
void HCSR04::start(void)
{
    trigger.write(1);
    //printf("trigger before %d\n", trigger);
    wait_us(10);
    trigger.write(0);
   // printf("trigger after %d\n", trigger);
}

void HCSR04::isr_fall(void)
{
    pulsetime.stop();
    pulsedur = pulsetime.read_us();
    distance= (pulsedur*343)/20000;
    pulsetime.reset();
}

void HCSR04::rise (void (*fptr)(void))
{
    echo.rise(fptr);
}
void HCSR04::fall (void (*fptr)(void))
{
    echo.fall(fptr);
}

unsigned int HCSR04::get_dist_cm()
{
    return distance;
}
unsigned int HCSR04::get_pulse_us()
{
    return pulsedur;
}



/*******************************************************
   Here is a sample code usage
********************************************************* 
#include "hcsr04.h"
HCSR04  usensor(p25,p6);
int main()
{
    unsigned char count=0;
    while(1) {
        usensor.start();
        wait_ms(500); 
        dist=usensor.get_dist_cm();
        lcd.cls();
        lcd.locate(0,0);
        lcd.printf("cm:%ld",dist );
 
        count++;
        lcd.locate(0,1);
        lcd.printf("Distance =%d",count);
        
    }
*/