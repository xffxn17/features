# let's see how auto message is send in whatsapp using python
import schedule
import time
import datetime
import pywhatkit as kit

# let's see how auto message is send in whatsapp using python


def send_message():
    phone_number = "+918688584029"
    message = "Hello, This is an automated message."

    kit.sendwhatmsg_instantly(phone_number, message,
                              wait_time=15, tab_close=True)


schedule.every().day.at("00:03:00").do(send_message)
print("Waiting for schedule time...")
while True:
    schedule.run_pending()
    time.sleep(1)
