import numpy as np
import botor
import matplotlib.pyplot as plt
import pandas as pd
import os
import rebound


def read_file(path, file):
    """Liest eine einzelne Datei ein (egal in welchem unserer Formate [theoretisch]) und gibt ein dictionary zurück, 
    welches in "data"die Daten aus der Datei in einer pandas Series speichert und außerdem Felder für die verschiedenen
    Initialisierungsparameter hat wie "e_p", "e_bin", etc.
    """
    try:
        if file.endswith('.acrit'):
            # .acrit Dateien geben momentan noch einen "Status" aus, der angibt, weshalb die Simulation abgebrochen wurde.
            # Der wird aber nicht verwendet, weshalb er hier durch ".t_surv.squeeze()" weggelassen wird.
            data = {"data": pd.read_csv(path + file, index_col=0, names=["t_surv", "res"],
                                        sep=";", header=1).t_surv.squeeze()}
        elif file.endswith('.sim'):
            data = {"data": pd.read_csv(path + file, index_col=0, sep=";", header=1)}
            data["data"].columns = [data["data"].columns[k].strip() for k in range(len(data["data"].columns))]
        elif file.endswith('.bin') or file.endswith(".sa"):
            data = rebound.SimulationArchive(path + file)
    except:
        print(path + file)

    if not (file.endswith('.bin') or file.endswith('.sa')):
        # Gibt der "data"-Series als Namen den Wert, der im Dateinamen enthalten war
        # (nützlich zur einfachen Zuweisung, wenn später mehrere Dateien eingelesen werden)
        if ('.'.join(file.split('.')[:-1])).endswith("pi"):
            data["data"].name = np.pi*float('.'.join(file.split('=')[1].split('.')[0:2])[:-2])
        else:
            data["data"].name = float('.'.join(file.split('=')[1].split('.')[0:2]))

        f = open(path + file)
        header = f.readline()
        f.close()

        params = header.split(',')
        # Liest die verschiedenen Initialisierungsparameter ein und speichert sie im dictionary
        for i in range(len(params)):
            data[params[i].split('=')[0].strip()] = float(params[i].split('=')[1].split('*')[0])

    return data


def read_folder(path):
    """Liest einen ganzen Ordner von .acrit-Dateien ein und speichert die jeweils zurückgegebenen Daten in einer
    pandas Series, die als Index die durch die Dateinamen spezifizierten Werte besitzt.
    """

    stuff = os.listdir(path)
    indeces = []
    datas = []
    name = ''
    for file in stuff:
        if os.path.isdir(os.path.join(path, file)) or not (file.endswith('.acrit') or file.endswith('.sim')) \
                or file.endswith('.bin'):
            # Überspringt eventuelle Unterordner und andere Dateien
            continue
        else:
            # Liest den Wert aus dem Dateinamen aus
            # if file.endswith('pi.acrit') or file.endswith('pi.sim') or file.endswith('pi.sim'):
            if file.split('.')[-2][-2:] == "pi":
                indeces.append(np.pi*float('.'.join(file.split('=')[1].split('.')[0:2])[:-2]))
            else:
                indeces.append(float('.'.join(file.split('=')[1].split('.')[0:2])))
            datas.append(read_file(path, file))
            # Stellt fest, welcher Parameter zwischen den gegebenen Dateien variiert wurde
            # (Wird später der Series als name übergeben)
            name = (file.split('=')[0])

    # Sortiert die eingelesenen Daten anhand der Werte in den Dateinamen
    sort_index = np.argsort(indeces)
    return_data = pd.Series([datas[k] for k in sort_index], [indeces[k] for k in sort_index], name=name)

    return return_data


def read_acrit_calc(path, dim):
    """ Liest eine Ordnerstruktur aus .acrit Daten, die mehr als eine Ebene besitzt, ein.
    Wird verwendet, wenn mehrere Parameter variiert werden und ist rekursiv implementiert, um unabhängig von der
    Tiefe der Ordnerstruktur verwendet werden zu können (Tiefe muss in "dim" übergeben werden).

    Resultat ist eine Series, die wiederum Series' enthält, etc. bis letztendlich das dictionary einer einzelnen
    Datei erreicht ist.
    """

    indices = []
    name = ''
    stuff = os.listdir(path)
    for thing in stuff:  # 1A Variablenbezeichnung
        if dim > 1 and not os.path.isdir(os.path.join(path, thing)):
            # Sollte eventuell später angepasst werden, falls in tiefe Ordnerstrukturen noch info-Dateien hinzugefügt
            # werden.
            print(os.path.join(path, thing))
            print("There are files where only folders should be, what did you do?")
            return
        # Sammelt die in der aktuellen Ebene vorhandenen Parameter-Werteif ('.'.join(file.split('.')[:-1])).endswith("pi"):
        #     data["data"].name = np.pi*float('.'.join(file.split('=')[1].split('.')[0:2])[:-2])
        # else:
        #     data["data"].name = float('.'.join(file.split('=')[1].split('.')[0:2]))
        if not thing.endswith("pi"):
            indices.append(float(thing.split("=")[1]))
        else:
            indices.append(float(thing.split("=")[1][:-2]))
        name = thing.split('=')[0]  # Stellt fest, welcher Parameter variiert wurde
    # Sortiert die eingelesenen Daten anhand der Werte in den Dateinamen
    sort_index = np.argsort(indices)

    # Geht rekursiv in die Ordnerstruktur runter, wenn nötig
    if dim == 1:
        data = []
        for folder in [stuff[k] for k in sort_index]:
            data.append(read_folder(path + folder + "/"))
        return_data = pd.Series(data, [indices[k] for k in sort_index], name=name)
    elif dim > 1:
        data = []
        for folder in [stuff[k] for k in sort_index]:
            data.append(read_acrit_calc(path + folder + "/", dim - 1))
        return_data = pd.Series(data, [indices[k] for k in sort_index], name=name)
    else:
        print("What are you doing with strange dimensions?")
        return_data = []

    return return_data

