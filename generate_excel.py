import xlsxwriter
import os

def create_retirement_model(filename):
    workbook = xlsxwriter.Workbook(filename)
    
    # Formaty
    fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
    fmt_input_label = workbook.add_format({'bold': True, 'align': 'right'})
    fmt_input_val = workbook.add_format({'bg_color': '#E2EFDA', 'border': 1, 'num_format': '#,##0.00'})
    fmt_input_pct = workbook.add_format({'bg_color': '#E2EFDA', 'border': 1, 'num_format': '0.00%'})
    fmt_input_int = workbook.add_format({'bg_color': '#E2EFDA', 'border': 1, 'num_format': '0'})
    
    fmt_currency = workbook.add_format({'num_format': '#,##0', 'border': 1})
    fmt_pct = workbook.add_format({'num_format': '0.00%', 'border': 1})
    fmt_int = workbook.add_format({'num_format': '0', 'border': 1})
    fmt_text = workbook.add_format({'align': 'center', 'border': 1})
    
    # --- Arkusz: Parametry ---
    ws_params = workbook.add_worksheet("Parametry")
    ws_params.set_column('A:A', 35)
    ws_params.set_column('B:B', 20)
    
    ws_params.write('A1', 'Parametry Modelu Emerytalnego', fmt_header)
    ws_params.write('B1', 'Wartość', fmt_header)
    
    params = [
        ('Kapitał Początkowy (PLN)', 1000000, fmt_input_val),      # B2
        ('Wydatki Miesięczne dzisiaj (PLN)', 5000, fmt_input_val), # B3
        ('ZUS Miesięczny dzisiaj (PLN)', 2000, fmt_input_val),     # B4
        ('Wpłaty Miesięczne przed emeryturą (PLN)', 3000, fmt_input_val), # B5
        ('Inflacja Bazowa Roczna', 0.03, fmt_input_pct),           # B6
        ('Oczekiwany Zwrot Roczny', 0.07, fmt_input_pct),          # B7
        ('Wiek Obecny', 53, fmt_input_int),                        # B8
        ('Wiek Emerytalny', 60, fmt_input_int),                    # B9
        ('Horyzont (Zakładany Wiek Końcowy)', 95, fmt_input_int),  # B10
        ('Podatek od zysków kapitałowych (np. 19% lub 0% IKE)', 0.19, fmt_input_pct) # B11
    ]
    
    for i, (label, val, fmt) in enumerate(params):
        row = i + 1
        ws_params.write(row, 0, label, fmt_input_label)
        ws_params.write(row, 1, val, fmt)
        
    # Instrukcja
    ws_params.merge_range('A14:B14', 'Zmień wartości w zielonych komórkach, aby zaktualizować model.', workbook.add_format({'italic': True, 'font_color': 'gray'}))

    # --- Arkusz: Model ---
    ws_model = workbook.add_worksheet("Model")
    
    headers = [
        "Rok symulacji", "Wiek", "Faza", "Mnożnik Inflacji", 
        "Kapitał na początku roku", "Roczne Wpłaty netto", 
        "Roczne ZUS", "Wymagane Wydatki Brutto", "Wypłata z portfela", 
        "Zysk Rynkowy (Brutto)", "Baza Kosztowa (do podatku)", "Podatek Belki", 
        "Kapitał na koniec roku"
    ]
    for col, h in enumerate(headers):
        ws_model.write(0, col, h, fmt_header)
        
    ws_model.set_column('A:B', 12)
    ws_model.set_column('C:C', 15)
    ws_model.set_column('D:D', 18)
    ws_model.set_column('E:M', 22)
    
    # Param names for formulas
    p_cap_init = "Parametry!$B$2"
    p_exp_mo = "Parametry!$B$3"
    p_zus_mo = "Parametry!$B$4"
    p_dep_mo = "Parametry!$B$5"
    p_inf = "Parametry!$B$6"
    p_ret = "Parametry!$B$7"
    p_age_curr = "Parametry!$B$8"
    p_age_ret = "Parametry!$B$9"
    p_age_end = "Parametry!$B$10"
    p_tax = "Parametry!$B$11"
    
    # Write 60 years of simulation logic (up to age ~110)
    # The max rows = 110 - 18 = 92 rows. Let's do 100 rows.
    MAX_ROWS = 100
    
    for i in range(MAX_ROWS):
        row = i + 1
        r_excel = row + 1 # 1-indexed for excel formulas
        
        # A: Rok (1, 2, 3...)
        ws_model.write(row, 0, f'=IF({p_age_curr}+{row}-1<={p_age_end}, {row}, "")', fmt_int)
        
        # B: Wiek
        ws_model.write(row, 1, f'=IF(A{r_excel}="","", {p_age_curr}+A{r_excel}-1)', fmt_int)
        
        # C: Faza
        ws_model.write(row, 2, f'=IF(A{r_excel}="","", IF(B{r_excel}>={p_age_ret}, "Dekumulacja", "Akumulacja"))', fmt_text)
        
        # D: Mnożnik Inflacji
        if i == 0:
            ws_model.write(row, 3, f'=IF(A{r_excel}="","", 1)', fmt_pct)
        else:
            ws_model.write(row, 4 - 1, f'=IF(A{r_excel}="","", D{r_excel-1}*(1+{p_inf}))', fmt_pct)
            
        # E: Kapitał na początku roku
        if i == 0:
            ws_model.write(row, 4, f'=IF(A{r_excel}="","", {p_cap_init})', fmt_currency)
        else:
            ws_model.write(row, 5 - 1, f'=IF(A{r_excel}="","", M{r_excel-1})', fmt_currency)
            
        # F: Roczne Wpłaty netto
        ws_model.write(row, 5, f'=IF(A{r_excel}="","", IF(C{r_excel}="Akumulacja", {p_dep_mo}*12*D{r_excel}, 0))', fmt_currency)
        
        # G: Roczne ZUS
        ws_model.write(row, 6, f'=IF(A{r_excel}="","", IF(C{r_excel}="Dekumulacja", {p_zus_mo}*12*D{r_excel}, 0))', fmt_currency)
        
        # H: Wymagane Wydatki Brutto
        ws_model.write(row, 7, f'=IF(A{r_excel}="","", IF(C{r_excel}="Dekumulacja", {p_exp_mo}*12*D{r_excel}, 0))', fmt_currency)
        
        # I: Wypłata z portfela
        ws_model.write(row, 8, f'=IF(A{r_excel}="","", MAX(0, H{r_excel}-G{r_excel}))', fmt_currency)
        
        # J: Zysk Rynkowy Brutto
        ws_model.write(row, 9, f'=IF(A{r_excel}="","", MAX(0, E{r_excel}+F{r_excel}-I{r_excel})*{p_ret})', fmt_currency)
        
        # K: Baza Kosztowa
        if i == 0:
            # First year: initial cap + deposits - proportional withdrawals
            ws_model.write(row, 10, f'=IF(A{r_excel}="","", MAX(0, {p_cap_init} + F{r_excel} - I{r_excel} * MIN(1, ({p_cap_init}) / MAX(1, E{r_excel}+F{r_excel}))))', fmt_currency)
        else:
            ws_model.write(row, 10, f'=IF(A{r_excel}="","", MAX(0, K{r_excel-1} + F{r_excel} - I{r_excel} * MIN(1, K{r_excel-1} / MAX(1, E{r_excel}+F{r_excel}))))', fmt_currency)
            
        # L: Podatek Belki (z wypłat proporcjonalnie do udziału zysku)
        if i == 0:
            ws_model.write(row, 11, f'=IF(A{r_excel}="","", IF({p_tax}>0, I{r_excel} * (1 - MIN(1, ({p_cap_init}) / MAX(1, E{r_excel}+F{r_excel}))) * {p_tax}, 0))', fmt_currency)
        else:
            ws_model.write(row, 11, f'=IF(A{r_excel}="","", IF({p_tax}>0, I{r_excel} * (1 - MIN(1, K{r_excel-1} / MAX(1, E{r_excel}+F{r_excel}))) * {p_tax}, 0))', fmt_currency)
        
        # M: Kapitał Końcowy
        ws_model.write(row, 12, f'=IF(A{r_excel}="","", MAX(0, E{r_excel}+F{r_excel}-I{r_excel}+J{r_excel}-L{r_excel}))', fmt_currency)


    # --- Arkusz: Wykres ---
    ws_chart = workbook.add_worksheet("Wykres")
    
    chart = workbook.add_chart({'type': 'line'})
    
    # Serie dla wykresu kapitału
    chart.add_series({
        'name':       'Kapitał Końcowy',
        'categories': ['Model', 1, 1, MAX_ROWS, 1], # X-axis: Wiek
        'values':     ['Model', 1, 12, MAX_ROWS, 12], # Y-axis: Kapitał końcowy
        'line':       {'color': '#4472C4', 'width': 2.25},
    })
    
    chart.set_title({'name': 'Symulacja Kapitału Emerytalnego (Model Deterministyczny)'})
    chart.set_x_axis({'name': 'Wiek', 'major_gridlines': {'visible': True}})
    chart.set_y_axis({'name': 'Kapitał (PLN)', 'major_gridlines': {'visible': True}, 'num_format': '#,##0'})
    chart.set_legend({'position': 'bottom'})
    chart.set_size({'width': 800, 'height': 450})
    
    ws_chart.insert_chart('B2', chart)
    
    workbook.close()
    print(f"File {filename} created successfully.")

if __name__ == "__main__":
    create_retirement_model("Emerytura_Model.xlsx")
