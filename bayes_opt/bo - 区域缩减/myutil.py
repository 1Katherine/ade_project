import time
from datetime import datetime, timedelta

class myutil:
    def format_time(t):
        # if time < 60:
        #     print(str(time) + 's')
        # elif time > 60 and time < 3600:
        #     second = int(time)
        #     minute = divmod(millis, 1000)
        #     print(str(time/60) + 'm' + str(time%60) + 's')
        # else:
        #     second = int(time)
        #     minute = time / 60 % 60
        #     hour = time / (60 * 60) %
        #     print(str(time/3600) + 'h' + str(time%60) + 'm' + str(time%60%60) + 's')

        minutes, seconds = divmod(t, 60)
        hours, minutes = divmod(minutes, 60)
        return ("%d:%d:%d" % (hours, minutes, seconds))

# print(time.time())


if __name__ == '__main__':

    # 记录开始时间
    start = time.time()
    print(start)


    # 定时2s
    time.sleep(2)
    # 记录结束时间
    end = time.time()
    print(end)

    al_duration = end - start
    # print(end - start)
    print(al_duration)  # 获取当前时间精确到秒数


    print (str(int(al_duration)) + 's')                  #秒级时间戳
    # print (int(round(al_duration * 1000)))    #毫秒级时间戳

    util = myutil()
    util.format_time(al_duration)

