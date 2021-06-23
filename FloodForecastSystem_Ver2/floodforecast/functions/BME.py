# from ..data.rainstation_data import _stationData
import os,twd97,json
import pandas as pd
import numpy as np
from floodforecast.data.rainstation_data import _stationData
from datetime import datetime, timedelta
from BMEFunction import BMEestimation

class BME:
    def __init__(self, bmeObsRainDict,estTlen = 3):
        
        self.bmeObsRainDict = bmeObsRainDict
        # get estimated station name
        self.stationNameList = list(bmeObsRainDict.keys())
        # get estimated Timestamp
        self.timelist = pd.DataFrame(bmeObsRainDict[self.stationNameList[0]])['time'].tolist()
        self.estTlen = estTlen
        self.csvPath = os.path.join(os.getcwd(), 'floodforecast', 'data', 'csv')
        
    def BMEformatter(self, dataframe, points):
        lng84Series = dataframe.loc[points].iloc[:, 0]
        lat84Series = dataframe.loc[points].iloc[:, 1]
        pdict_v = dataframe.loc[points].iloc[:, 2]
        
        lng97List = []
        lat97List = []
        for lat, lng in zip(lat84Series, lng84Series):
            x, y = twd97.fromwgs84(lat, lng)
            lng97List.append(x)
            lat97List.append(y)
        lngSeries = pd.Series(lng97List)
        latSeries = pd.Series(lat97List)

        return lngSeries, latSeries,pdict_v
    
    def GetBMESimInput(self,Stacode):
        # get forcasting value and location by given grid points and CSV file
        points = _stationData[Stacode]['points']
        ObsValue = pd.DataFrame(self.bmeObsRainDict[Stacode])['rainfall'].tolist()
        StaPdict = pd.DataFrame([])
        T = 1 
        for i,j in zip(self.timelist,ObsValue):
            GetDataTimeFormat = pd.Timestamp(i).strftime('%Y%m%d%H')
            GetDataTimePath = os.path.join(self.csvPath,(GetDataTimeFormat+'.csv'))
            dataframe = pd.read_csv(GetDataTimePath)
            x, y , pdict_v = self.BMEformatter(dataframe=dataframe, points=points)
            aStaPdict = pd.DataFrame(np.vstack((x.values,y.values,T*np.ones(x.shape),
                                j*np.ones(x.shape),pdict_v.values)).T,index = points)
            StaPdict = pd.concat((StaPdict,aStaPdict),axis = 0)
            T+=1
        StaPdict.columns = ['X','Y','T','Z_obs','Z_p']
        StaPdict = StaPdict.assign(Z = StaPdict['Z_obs']-StaPdict['Z_p'])
        return StaPdict
    
    def CreatGridInput(self,StaPdict):
        # creat grid
        pointsxydf = StaPdict[['X','Y']].drop_duplicates().reset_index()
        pointsxydf.columns = ['points','X','Y']
        tME = np.arange(len(self.timelist)+1,len(self.timelist)+1+self.estTlen)
        gridxy = np.tile(pointsxydf[['X','Y']].values,(len(tME),1))
        gridt = np.repeat(tME,len(pointsxydf))
        return gridxy[:,0],gridxy[:,1],gridt

    def BMEpostprocess(self,StaPdict,BMEresult):

        ## BME output arrangement
        pointsxydf = StaPdict[['X','Y']].drop_duplicates().reset_index()
        pointsxydf.columns = ['points','X','Y']
        BMEresult = BMEresult.merge(pointsxydf,on = ['X','Y'])
        earliestTime = self.timelist[0]
        estT = [ str(pd.Timestamp(earliestTime)+timedelta(hours=int(i)-1, minutes=0)) for i in BMEresult['T']]
        BMEresult = BMEresult.assign(realT = estT).sort_values(by=['T','points'],ascending= [True,True])
        BMEresult = BMEresult.set_index(BMEresult.pop('points'))
        
        ## BME input arrangement
        StaPdict = StaPdict.reset_index()
        StaPdict.columns = ['points']+StaPdict.columns[1:].tolist()
        rawT = [ str(pd.Timestamp(earliestTime)+timedelta(hours=int(i)-1, minutes=0)) for i in StaPdict['T']]
        StaPdict = StaPdict.assign(realT = rawT).reset_index().sort_values(by=['T','points'],ascending= [True,True])
        StaPdict = StaPdict.set_index(StaPdict.pop('points'))
        
        return StaPdict,BMEresult
    
    def BMEprocess(self,Detrendmethod = 0,maxR = None, nrLag = None, rTol = None, 
            maxT = 3, ntLag = 3, tTol = 1.5, EmpCv_parashow = False, EmpCv_picshow = False,
            CVfit_Sinit_v=None, CVfit_Tinit_v=3, CVfit_plotshow = False,
            BME_nhmax=None,BME_nsmax=None,BME_dmax=None):
        
        BEMinputdict = {}
        BMEoutputdict = {}
        for Stacode in self.stationNameList:
            
            ## BME preparation
            StaPdict = self.GetBMESimInput(Stacode)
            estX,estY,estT = self.CreatGridInput(StaPdict)
            Points = StaPdict[['X','Y','T']].values# shape must be n*3
            Z = StaPdict[['Z']].values.reshape(-1,1) # shape must be n*1
            EstPoints = np.hstack((estX.reshape(-1,1),estY.reshape(-1,1),estT.reshape(-1,1)))
            
            ## create estimate class
            BMEobject = BMEestimation(Points,Z,EstPoints,DetrendMethod=Detrendmethod)
            
            ## calculate emperical covariance
            BMEobject.Empirical_covplot(maxR = maxR, nrLag = nrLag, rTol = rTol, 
                                        maxT = maxT, ntLag = ntLag, tTol = tTol,
                                        parashow = EmpCv_parashow, picshow = EmpCv_picshow)
            
            # Covariance model autofitting
            covmodel,covparam = BMEobject.Covmodelfitting(Sinit_v = CVfit_Sinit_v,
                                                          Tinit_v = CVfit_Tinit_v,
                                                          plotshow = CVfit_plotshow)
            
            ## BME estimation
            BMEresult = BMEobject.BMEestimationH(nhmax = BME_nhmax,
                                                 nsmax = BME_nsmax,
                                                 dmax = BME_dmax)
            
            ## BME result postprocess
            BMEinput,BMEoutput = self.BMEpostprocess(StaPdict,BMEresult)
            
            ## save result to dictionay
            BMEoutputdict.update({Stacode:BMEoutput})
            
            ## save input data to dictionay
            BEMinputdict.update({Stacode:BMEinput})
            
        return BEMinputdict,BMEoutputdict
        
if __name__ == '__main__':
    
    pass

