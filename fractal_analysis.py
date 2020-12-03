import talib
import yfinance as yf
import pandas as pd
import pprint


class func_results:
    def __init__(self):
        self.dictmap = {}
        self.engulfing = 0
        self.morningstar = 0
        self.shootingstar = 0
        self.doji = 0
        self.dojistar = 0
        self.threeinside = 0
        self.threeblackcrows = 0
        self.abandonedbaby = 0
        self.hammer = 0
        self.invertedhammer = 0


    def CDLENGULFING(self, engulfing):
        self.engulfing = engulfing

    def CDLMORNINGSTAR(self, morningstar):
        self.morningstar = morningstar

    def CDLSHOOTINGSTAR(self, shootingstar):
        self.shootingstar = shootingstar

    def CDLDOJI(self, doji):
        self.doji = doji

    def CDLDOJISTAR(self, dojistar):
        self.dojistar = dojistar

    def CDL3INSIDE(self, threeinside):
        self.threeinside = threeinside

    def CDL3BLACKCROWS(self, threeblackcrows):
        self.threeblackcrows = threeblackcrows

    def CDLABANDONEDBABY(self, abandonedbaby):
        self.abandonedbaby = abandonedbaby

    def CDLHAMMER(self, hammer):
        self.hammer = hammer

    def CDLINVERTEDHAMMER(self, invertedhammer):
        self.invertedhammer = invertedhammer



    def __repr__(self):
        return 'func_results(' + repr(self.engulfing) + ', ' + repr(self.morningstar) + ', ' + repr(self.shootingstar) + ', ' + repr(self.doji) + ', ' + repr(self.dojistar) + ', ' + repr(self.threeinside) + ', ' + repr(self.threeblackcrows) + ', ' + repr(self.abandonedbaby) + ', ' + repr(self.hammer) + ', ' + repr(self.invertedhammer) + ')'


result_dict = dict()

data = yf.download("[TICKER]", start="2020-10-01", end="2020-12-02")
methods = ["CDLENGULFING", "CDLMORNINGSTAR", "CDLSHOOTINGSTAR", "CDLDOJI", "CDLDOJISTAR"]
result_days = {}
final_days={"CDLENGULFING" : [], "CDLMORNINGSTAR" : [], "CDLSHOOTINGSTAR": [], "CDLDOJI": [], "CDLDOJISTAR": [], "CDL3INSIDE": [], "CDL3BLACKCROWS": [], "CDLABANDONEDBABY": [], "CDLHAMMER": [], "CDLINVERTEDHAMMER": []}
for methodname in methods:
    print(methodname)
    method = getattr(talib, methodname)
    data[methodname] = method(data['Open'], data['High'], data['Low'], data['Close'])
    result_days[methodname] = data[data[methodname] != 0]
    print(result_days[methodname])
for methodname in methods:
    dates = result_days[methodname]
    meth = getattr(func_results, methodname)
    for i, j in dates.iterrows():
        res = result_dict.get(i)
        if res is None:
            func = func_results()
            meth(func, j[methodname])
            result_dict[i]=func
        else:
            meth(res, j[methodname])

df = pd.DataFrame(data=result_dict, index=[0])
df = (df.T)
print (df)


pp = pprint.PrettyPrinter(indent=4)
pp.pprint(result_dict)
print(df).to_excel("output.xlsx")
