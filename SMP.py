#=======================================================Libraries==============================================================#
#calculation and manipulation
import pandas as pd
import datetime as dt
import math
import numpy as np

#Machine Learning Algorithms
from sklearn import preprocessing, svm
import sklearn.model_selection as skms
from sklearn.linear_model import LinearRegression

#Plotting
import matplotlib.pyplot as plt
from matplotlib import style

#For GUI
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


#matplotlib.use("TKAGG")
style.use('bmh')

#===========================================================GUI================================================================#

#NOS stands for Name Of Stock
NOS = ''
direction = 1

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 600, height = 500, relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='Stock Market Predictor')
label1.config(font=('helvetica', 26))
canvas1.create_window(320, 25, window=label1)

label2 = tk.Label(root, text='Name of Stock/Ticker Symbol (In Capital. E.g. FB, TSLA, GOOGL) :')
label2.config(font=('helvetica',14))
canvas1.create_window(320, 180, window=label2)

label11 = tk.Label(root, text='*The API used for Stocks\nin NSE is fairly new \n**Might Crash**')
label11.config(font=('helvetica',9))
canvas1.create_window(500, 450, window=label11)

entry1 = tk.Entry (root,width=35) 
canvas1.create_window(320, 220, window=entry1)


def _quit():
    #quiting and destroying to save memory resources
    tk.messagebox.showinfo('Processing','System may take a while to Compute!')
    root.quit()     
    root.destroy()   

#Aditional Options Displayed    
def Yes_Selected():

    #clearing canvas to write on that same space
    canvas1.delete('all')

    label3 = tk.Label(root, text='What do you think will be the \nGeneral direction for the next Trading Day?')
    label3.config(font=('helvetica',13))
    canvas1.create_window(320, 100, window=label3)
    
    #used to see which Radiobutton was choosen
    var = tk.IntVar()
    var.set(1)

    R1 = tk.Radiobutton(root, text="Upwards", variable=var, value=1,activeforeground='blue')
    canvas1.create_window(312, 140, window=R1)

    R2 = tk.Radiobutton(root, text="Downwards", variable=var, value=2,activeforeground='blue')
    canvas1.create_window(320, 170, window=R2)

    button_next = tk.Button(root, text='Next', command=_quit, bg='#0073e6', fg='white', font=('helvetica', 11))
    canvas1.create_window(320, 350, window=button_next)
    
    label_high = tk.Label(root, text='*Optional')
    label_high.config(font=('helvetica',9))
    canvas1.create_window(210, 190, window=label_high)
    
    entry_high = tk.Entry (root,width=25) 
    canvas1.create_window(320, 220, window=entry_high)
    label_high = tk.Label(root, text='High')
    label_high.config(font=('helvetica',11))
    canvas1.create_window(220, 220, window=label_high)
    
    entry_open = tk.Entry (root,width=25) 
    canvas1.create_window(320, 260, window=entry_open)
    label_open = tk.Label(root, text='Open')
    label_open.config(font=('helvetica',11))
    canvas1.create_window(220, 260, window=label_open)
    
    entry_close = tk.Entry (root,width=25) 
    canvas1.create_window(320, 300, window=entry_close)
    label_close = tk.Label(root, text='Close')
    label_close.config(font=('helvetica',11))
    canvas1.create_window(220, 300, window=label_close)

#Window for user to choose Additional options or not   
def Include_Intuition():

    canvas1.delete('all')
    label_ = tk.Label(root, text='Do you want to Include Additional Options/Include Intuition ?')
    label_.config(font=('helvetica',15))
    canvas1.create_window(320, 200, window=label_)

    button_y = tk.Button(root, text='YES', command=Yes_Selected, bg='#0073e6', fg='white', font=('helvetica', 11, 'bold'))
    canvas1.create_window(250, 260, window=button_y)
    button_n = tk.Button(root, text='NO', command=_quit, bg='#ff9933', fg='white', font=('helvetica', 11, 'bold'))
    canvas1.create_window(340, 260, window=button_n)

#Intial Name of the Stock Entered   
def take_input():

    global NOS
    NOS = entry1.get()
    Include_Intuition()
    #root.quit()
    #root.destroy() 

    
button1 = tk.Button(root, text='GO', command=take_input, bg='green', fg='white', font=('helvetica', 11, 'bold'))
canvas1.create_window(320, 260, window=button1)
root.mainloop()



#======================================================Prediction_Model========================================================#

#interval for data gathering
start = dt.datetime(2000,1,1)
end = dt.datetime.now()

df = pd.DataFrame()

##check if NOS data is already present or not
try:
    df = read_csv(NOS+'.csv')
except:
    #print(1)
    pass

#For importing stock data from different indexes
#Fetching data from different APIs
try:
    import pandas_datareader.data as web
    df = web.DataReader(NOS, 'yahoo', start, end)
except:
    #print(2)
    try:
        from nsepy import get_history
        df = get_history(NOS, start, end)
        
    except:
        #print(3)
        try:
            import yfinance as yf
            data = yf.download(NOS, start, end)
        except:
           # print(4)
            tk.messagebox.showwarning('Error!','System was not able to find data for the Entered Stock')
            exit()
            

#save Data of NOS for future Use
df.to_csv(NOS+'.csv')


#volatility;;percentage change
df['HL_PCT'] = ( (df['High']-df['Close']) / df['Close'] ) * 100
df['PCT_change'] = ( (df['Close']-df['Open']) / df['Open'] ) * 100

#needed attributes
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
df_main = df
forecast_col = 'Close'
df.fillna(-99999,inplace = True) #making them an outlier

#days we are forecasting forward
forecast_out = int(math.ceil(0.01*len(df))) ##last ten days

#print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

#features apart from label;;CAPITAL X
X = np.array(df.drop(['label'],1)) #returns a new df
X = preprocessing.scale(X)
X = X[:-forecast_out]


X_lately = X[-forecast_out:] ########new data to test against


#Last values have NA so we will drop them
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

#Test-Train Split
X_train, X_test, y_train, y_test = skms.train_test_split(X,y,test_size = 0.3)

#Linear Regression classifier
clf = LinearRegression(n_jobs = -1) #runs as many jobs at a time as possible
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
#print(accuracy)

acc = accuracy

#SVR classifier
clf2 = svm.SVR()
clf2.fit(X_train,y_train)
accuracy2 = clf2.score(X_test,y_test)
#print(accuracy2)


#choosing the model with better accuracy for forecast
if accuracy > accuracy2 :
    forecast_set = clf.predict(X_lately)
else:
    forecast_set = clf2.predict(X_lately)
    acc = accurcy2


#print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

##for x-axis date;; since we droped it for calculation
last_date = df.iloc[-1].name
last_unix = 0
try: 
    last_unix = last_date.timestamp()
except:
    import time
    last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]  
    
#100 rolling moving average	
df['100ma'] = df_main['Close'].rolling(window=100,min_periods = 0).mean()   ##moving average;;intial elements will have nan   

df.to_csv(NOS+'_Predicted.csv')

#========================================================Graph=================================================================#
#GUI

df2 = df
root = tk.Tk()
#root.wm_title("Embedding in Tk")

fig = Figure(figsize=(10, 5), dpi=100)
ax2 = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.

#Func when Quit is pressed
def quit2(root):

    #global root
    root.quit()
    root.destroy()

#To display Predicted Graph of the forecasted values    
def predicted_graph():

    #global root
    root = tk.Tk()
    #root.wm_title("Embedding in Tk")

    fig = Figure(figsize=(10, 5), dpi=100)
    ax2 = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    #ploting
    #Prediction
    pricemin = df2['Close'].min()
    df2['Close'].plot(kind='line', linewidth = .8,legend =True,ax = ax2, color = '#3385ff',fontsize = 10)
    df2['Forecast'].plot(kind='line', linewidth = .7,legend =True,ax = ax2, color = '#00e600',fontsize=10)
    ax2.fill_between(df2.index, pricemin, df['Forecast'], facecolor='#33ff33', alpha=0.5)
    ax2.fill_between(df2.index, pricemin, df['Close'], facecolor='#3385ff', alpha=0.5)

    #Navigation toolbar
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    #button = tk.Button(master=root, text="Quit", command=terminate, bg='#ff0000', fg='white', font=('helvetica', 11, 'bold'))
    #button.pack(side=tk.BOTTOM)

    button2 = tk.Button(master=root, text="Moving Average Curve", command=lambda:[quit2(root),moving_average_curve()], bg='#0066ff', fg='white', font=('helvetica', 11, 'bold'))
    button2.pack(side=tk.BOTTOM)

    label_acc = tk.Label(master=root, text='Accuracy Of Model : '+str(accuracy))
    label_acc.config(font=('helvetica',12))
    label_acc.pack(side=tk.RIGHT)

    root.title('Stock Market Predictor')
    ax2.set_title('Predicted Graph For '+ NOS,color = '#ffb31a')
	
    tk.mainloop()

#To display Moving average curve  
def moving_average_curve():
	
	#global root
    root = tk.Tk()
    #root.wm_title("Embedding in Tk")

    fig = Figure(figsize=(10, 5), dpi=100)
    ax2 = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.

    df_main['Close'].plot(kind='line', linewidth = .8,legend =True,ax = ax2, color = '#3385ff',fontsize = 10)
    df['100ma'].plot(kind='line', linewidth = .7,legend =True,ax = ax2, color = '#00e600',fontsize=10)

    #Navigation toolbar
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    #button = tk.Button(master=root, text="Quit", command=terminate, bg='#ff0000', fg='white', font=('helvetica', 11, 'bold'))
    #button.pack(side=tk.BOTTOM)

    button2 = tk.Button(master=root, text="Predicted Graph", command=lambda:[quit2(root),predicted_graph()], bg='#ff9933', fg='white', font=('helvetica', 11, 'bold'))
    button2.pack(side=tk.BOTTOM)
    
    label_acc = tk.Label(master=root, text='Current Moving Average : '+str(df['100ma'].iloc[-1]))
    label_acc.config(font=('helvetica',12))
    label_acc.pack(side=tk.RIGHT)

    root.title('Stock Market Predictor')
    ax2.set_title('Moving Average Curve for '+ NOS,color = '#ffb31a')
    tk.mainloop()

    
predicted_graph()

tk.mainloop()
root.quit()
root.destroy()
