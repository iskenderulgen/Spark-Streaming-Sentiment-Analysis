import pandas as pd
import settings
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from collections import Counter
def heat_g():
    mpl.style.use('ggplot')
    # print(settings.location)
    sehir_list = ['adana', 'adıyaman', 'afyon', 'ağrı', 'amasya', 'ankara', 'antalya', 'artvin',
                  'aydın', 'balıkesir', 'bilecik', 'bingöl', 'bitlis', 'bolu', 'burdur', 'bursa', 'çanakkale',
                  'çankırı', 'çorum', 'denizli', 'diyarbakır', 'edirne', 'elazığ', 'erzincan', 'erzurum', 'eskişehir',
                  'gaziantep', 'giresun', 'gümüşhane', 'hakkari', 'hatay', 'ısparta', 'mersin', 'istanbul', 'izmir',
                  'kars', 'kastamonu', 'kayseri', 'kırklareli', 'kırşehir', 'kocaeli', 'konya', 'kütahya', 'malatya',
                  'manisa', 'kahramanmaraş', 'mardin', 'muğla', 'muş', 'nevşehir', 'niğde', 'ordu', 'rize', 'sakarya',
                  'Samsun', 'Siirt', 'Sinop', 'Sivas', 'Tekirdağ', 'Tokat', 'Trabzon', 'Tunceli', 'Şanlıurfa', 'Uşak',
                  'van', 'yozgat', 'zonguldak', 'aksaray', 'bayburt', 'karaman', 'kırıkkale', 'batman', 'şırnak',
                  'bartın', 'ardahan', 'ığdır', 'yalova', 'karabük', 'kilis', 'osmaniye', 'düzce']

    list1 = []
    for i in settings.location:
        if i in sehir_list:
            list1.append(i)
    counts = Counter(list1)  #Eleman sayılarını bulma
    # print(counts)
    df = pd.DataFrame(list(counts.items()), columns=['sehir', 'sayac'])    #Dataframe oluşturma
    df['ulke'] = "Türkiye"
    print(df)
    df["ulke"] = pd.Categorical(df["ulke"], df.ulke.unique())
    df.head()
    matris = df.pivot("ulke", "sehir", "sayac")
    #print(matris)
    fig = plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    heatplot = ax.imshow(matris, cmap='Greens')
    ax.set_xticklabels(matris.columns)
    ax.set_yticklabels(matris.index)

    tick_spacing = 1.5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_title("Heatmap of Location")
    ax.set_xlabel('iller')
    
    plt.show()


#  sublist and dataframe
#  settings.location = [settings.location[i:i+1] for i in range(0,len(settings.location),1)]
#  df = pd.DataFrame(settings.location, columns=['sehir','ulke'])
#  print(settings.location)
#  print(df)
