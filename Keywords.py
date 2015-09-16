import pypyodbc
conn = pypyodbc.connect('Driver=FreeTDS;Server=ext.pitchbook.com;uid=pierce.young;pwd=fas44aca;database=dbd_copy')
print conn.cursor().execute('select top 10 * from news').fetchone()[0]
