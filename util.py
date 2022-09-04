from openpyxl import load_workbook


def output(date:str,
           col:str,
           row:str,
           val:int,
           datapath = './实验数据.xlsx'
           ):
    sheet_name = date + '实验数据'
    situation = col + row
    wb = load_workbook(datapath)
    ws1 = wb[sheet_name]
    ws1[situation].value = val
    cellB2_value = ws1['A2'].value
    wb.save("./实验数据.xlsx")
