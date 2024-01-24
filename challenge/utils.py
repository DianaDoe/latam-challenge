from datetime import datetime


def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff


def is_high_season(fecha):
    year = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=year)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=year)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=year)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=year)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=year)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=year)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=year)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=year)

    if ((range1_min <= fecha <= range1_max) or
            (range2_min <= fecha <= range2_max) or
            (range3_min <= fecha <= range3_max) or
            (range4_min <= fecha <= range4_max)):
        return 1
    else:
        return 0


def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()

    if morning_min < date_time < morning_max:
        return 'mañana'
    elif afternoon_min < date_time < afternoon_max:
        return 'tarde'
    elif (
            (evening_min < date_time < evening_max) or
            (night_min < date_time < night_max)
    ):
        return 'noche'

def get_data_balance(y_train):
    n_y0 = len(y_train[y_train == 0])
    n_y1 = len(y_train[y_train == 1])

    scale = n_y0 / n_y1
    return scale


    """
feature_important = model.get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)"""