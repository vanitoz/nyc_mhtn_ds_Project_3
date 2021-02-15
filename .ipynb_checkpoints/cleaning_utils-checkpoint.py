def compliance(row):
    """
    Checking if payment was made 
    before judgment_date or after 
    params : 
            row - each row in df
    """
    if row['payment_date'] < row['judgment_date']:
        return 1 
    else:
        return 0
    
