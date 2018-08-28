import datetime as dt

class Stepper(object):
    def __init__(self, prefix=''):
        """

        :type prefix: basestring
        """
        self.start = dt.datetime.now()
        self.ncalls = 0
        self.prefix = prefix
        self.write(msg="Start @ "+dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @staticmethod
    def formatTD(td):
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        return '%02d:%02d:%02d' % (hours, minutes, seconds)

    def new(self, msg='Step'):
        self.ncalls += 1
        print(self.prefix, self.ncalls, ". [", self.formatTD(dt.datetime.now()-self.start), "] ", msg)
        return dt.datetime.now()-self.start

    def write(self, msg):
        print ((10+len(self.prefix))*' '+msg)

    def end(self):
        return self.new(msg="The End @ "+dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

