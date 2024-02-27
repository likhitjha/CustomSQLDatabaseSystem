import ply.lex as lex
import ply.yacc as yacc
import pandas as pd
import numpy as np
import csv
import shutil
import os
import re
import pandas as pd
#pd.options.display.max_rows = None
np.random.seed(1)

# Define a database 20 rows and 3 columns
tokens = (
    'PRESENT',
    'IDENTIFIER',
    'THIS',
    'JOIN',
    'MENTIONED',
    'COLUMN_NAME',
    'TABLE_NAME',
    'NUMBER',
    'STRING',
    'LPAREN',
    'RPAREN',
    'EQUALS',
    'NOT_EQUALS',
    'GT',
    'LT',
    'COMMA',
    'SEMICOLON',
    'ASTERISK',  
    'BETWEEN',
    'LIKE',
    'IN',
    'IS_NULL',
    'AND',
    'OR',
    'NOT',
    'EXISTS',
    'UPPER',
    'LOWER',
    'TRIM',
    'CONCAT',
    'LEN',
    'SUBSTRING',
    'SUM',
    'MEAN',
    'COUNT',
    'LEAST',
    'MOST',
    'CLUSTER',
    'HAVING',
    'SORT',
    'STRAIGHT',
    'REVERSE',
    'LIMIT',
    'OFFSET',
    'UPDATE',
    'SET'
)

# Define a regular expression for identifiers (column names, table names, etc.)
def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value.upper(), 'IDENTIFIER')

    # Check if it's a reserved word, function, or '*'
    '''
    if t.type == 'IDENTIFIER':
        if column_state:
            t.type = 'COLUMN_NAME'
        else:
            t.type = 'TABLE_NAME'
    '''
    # Check if it's an asterisk '*'
    if t.value == '*':
        t.type = 'ASTERISK'

    if t.value.upper() == 'JOIN':
        t.type = 'JOIN'
    if t.value.upper() == 'SORT':
        t.type = 'SORT'
    if t.value.upper() == 'CLUSTER':
        t.type = 'CLUSTER'

    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_STRING(t):
    r'(\'[^\']*\'|\"[^\"]*\")'
    t.type = 'STRING'
    return t

# Define reserved words and functions
reserved = {
    'PRESENT': 'PRESENT',
    'THIS': 'THIS',
    'MENTIONED': 'MENTIONED',
    'BETWEEN': 'BETWEEN',
    'LIKE': 'LIKE',
    'IN': 'IN',
    'IS NULL': 'IS_NULL',
    'AND': 'AND',
    'OR': 'OR',
    'NOT': 'NOT',
    'EXISTS': 'EXISTS',
    'UPPER': 'UPPER',
    'LOWER': 'LOWER',
    'TRIM': 'TRIM',
    'CONCAT': 'CONCAT',
    'LEN': 'LEN',
    'SUBSTRING': 'SUBSTRING',
    'SUM': 'SUM',
    'MEAN': 'MEAN',
    'COUNT': 'COUNT',
    'LEAST': 'LEAST',
    'MOST': 'MOST',
    'CLUSTER': 'CLUSTER',
    'HAVING': 'HAVING',
    'SORT': 'SORT',
    'STRAIGHT': 'STRAIGHT',
    'REVERSE': 'REVERSE',
    'LIMIT': 'LIMIT',
    'OFFSET': 'OFFSET',
    'JOIN': 'JOIN', 
    'UPDATE':'UPDATE',
    'SET':'SET'
}

# Define SQL lexer rules
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_EQUALS = r'='
t_NOT_EQUALS = r'!='
t_GT = r'>'
t_LT = r'<'
t_COMMA = r','
t_SEMICOLON = r';'
t_JOIN = r'JOIN'
t_UPDATE = r'UPDATE'
t_SET = r'SET'

# Ignored characters (whitespace)
t_ignore = ' \t\n'

# Error handling rule
def t_error(t):
    print(f"Unexpected character: {t.value[0]}")
    t.lexer.skip(1)

# Initial state is column_state
column_state = True

def initialize_lexer():
    return lex.lex()

# Test the SQL to Python converter
def tokenize_sql_input(sql_input):
    lexer = initialize_lexer()
    lexer.input(sql_input)
    tokens = []
    global column_state
    while True:
        tok = lexer.token()
        if not tok:
            break
        if tok.type == 'THIS':
            column_state = True
        elif tok.type == 'MENTIONED':
            column_state = True
        elif tok.type == 'SEMICOLON':
            column_state = True
        tokens.append(tok)

    return tokens

def apply_projection(data_chunk, selected_columns):
    # Get column widths for data
    column_widths = [max(len(str(row[column])) for _, row in data_chunk.iterrows()) for column in selected_columns]
    index_width = max(len('Index'), len(str(len(data_chunk) - 1)))  # Width for the index column
    
    # Get column widths for column names
    column_name_widths = [max(len(column), column_widths[i]) for i, column in enumerate(selected_columns)]
    
    # Adjust column names for consistent spacing
    column_names = f"| {'Index':<{index_width}} |"
    for i, column in enumerate(selected_columns):
        column_names += f" {column:^{column_name_widths[i]}} |"
    print(column_names)
    
    # Print rows
    for index, row in data_chunk.iterrows():
        row_str = f"| {index:<{index_width}} |"
        for i, column in enumerate(selected_columns):
            row_str += f" {str(row[column]):^{column_name_widths[i]}} |"
        print(row_str)






def apply_single_condition(df, column, operator, value):
    if type(value)==str:
        value = value.strip("'")
    if operator == '=':
        return df[df[column] == value]
    elif operator == '!=':
        return df[df[column] != value]
    elif operator == '>=':
        return df[df[column] >= value]
    elif operator == '<=':
        return df[df[column] <= value]
    elif operator == '<':
        return df[df[column] < value]
    elif operator == '>':
        return df[df[column] > value]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def apply_conditions(df, conditions):
    """
    Apply conditions to the DataFrame and return the filtered DataFrame.
    """
    if not conditions:
        return df
    
    conditions_list = []
    conditions_list.append(conditions)


    for condition in conditions_list:
        if isinstance(condition, tuple) and len(condition) == 3 and (condition[0] != 'AND' and condition[0] != 'OR'):
            operator, column, value = condition
            df = apply_single_condition(df, column, operator, value)
        else:
            logical_operator = condition[0].lower()
            sub_condition1 = condition[1]
            sub_condition2 = condition[2]
            df_sub1 = apply_conditions(df.copy(), sub_condition1)
            df_sub2 = apply_conditions(df.copy(), sub_condition2)

            if logical_operator == 'and':
                df = pd.merge(df_sub1, df_sub2, how='inner', on=df.columns.tolist(), suffixes=('_left', '_right'))
            elif logical_operator == 'or':
                df = pd.concat([df_sub1, df_sub2]).drop_duplicates()

    return df


def apply_aggregate_function(aggregate_function, df):
    function, column = aggregate_function
    result = None

    if function == 'COUNT':
        result = df[column].count()
    elif function == 'SUM':
        result = df[column].sum()
    elif function == 'MEAN':
        result = df[column].mean()
    elif function == 'MOST':
        result = df[column].max()
    elif function == 'LEAST':
        result = df[column].min()

    return result

def apply_aggregate_functions(aggregate_functions, df):

    check1 = list(aggregate_functions)
    if type(check1[0]) == tuple:
        pass
    else:
        aggregate_functions= [tuple(aggregate_functions)]
    if isinstance(df, pd.core.groupby.DataFrameGroupBy):
        results = {}
        for group, group_df in df:
            results[group] = {}
            for func, col in aggregate_functions:
                
                results[group][f'{func}_{col}'] = apply_aggregate_function((func, col), group_df)
        
        # Convert dictionary to DataFrame for each group
        results_df = {group: pd.DataFrame(data=[results[group]]) for group in results}
        return pd.concat(results_df.values(), keys=results.keys()).reset_index(level=1, drop=True)
    else:
        results = {}
        
        for func, col in aggregate_functions:
                     
            results[f'{func}_{col}'] = apply_aggregate_function((func, col), df)
        
        return pd.DataFrame(data=results, index=[0])

def apply_group_by(df, columns):
    if columns:
        grouped = df.groupby(columns)
        return grouped
    return df



def get_csv_files_in_folder(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_files


def merge_sorted_csv(file1, file2, output_file, column_name, sort_order='STRAIGHT'):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w', newline='') as output:
        reader1 = csv.DictReader(f1)
        reader2 = csv.DictReader(f2)
        writer = csv.DictWriter(output, fieldnames=reader1.fieldnames)
        
        # Write headers to output file only if the file is empty
        if os.path.getsize(output_file) == 0:
            writer.writeheader()
        
        # Merge sorted rows
        row1 = next(reader1, None)
        row2 = next(reader2, None)
        
        while row1 and row2:
            try:
                val1 = float(row1[column_name])
                val2 = float(row2[column_name])
                if (sort_order == 'STRAIGHT' and val1 <= val2) or \
                   (sort_order == 'REVERSE' and val1 >= val2):
                    writer.writerow(row1)
                    row1 = next(reader1, None)
                else:
                    writer.writerow(row2)
                    row2 = next(reader2, None)
            except ValueError:
                # If conversion to float fails, use string comparison
                if (sort_order == 'STRAIGHT' and row1[column_name] <= row2[column_name]) or \
                   (sort_order == 'REVERSE' and row1[column_name] >= row2[column_name]):
                    writer.writerow(row1)
                    row1 = next(reader1, None)
                else:
                    writer.writerow(row2)
                    row2 = next(reader2, None)
        
        # Write remaining rows from reader1
        while row1:
            writer.writerow(row1)
            row1 = next(reader1, None)
        
        # Write remaining rows from reader2
        while row2:
            writer.writerow(row2)
            row2 = next(reader2, None)


# Example usage:
def merge_and_cleanup_files(column_name,sort_order):
   
    folder_path = './data/temp_sorted' 
    csv_files_list = get_csv_files_in_folder(folder_path)
  

    output_folder = './data/temp_sorted' 
    
 
    if len(csv_files_list) <= 1:
        merged_csv = os.path.join(folder_path, csv_files_list[0])
    else:
        merged_csv = os.path.join(folder_path, csv_files_list[0])
        for file in csv_files_list[1:]:
            file_path = os.path.join(folder_path, file)
            output_file = os.path.join(output_folder, f'merged_{file}')  # Unique output file for each iteration
            merge_sorted_csv(merged_csv, file_path, output_file, column_name, sort_order)
            merged_csv = output_file

    file_path = merged_csv
    result_file_path = './data/result/result.csv'
    
    directory_path = './data/result'
    
    for filename in os.listdir(directory_path):
        file = os.path.join(directory_path, filename)
        if os.path.isfile(file):
            os.remove(file)
            
    directory_path = './data/temp_sorted'

    shutil.copy(file_path, result_file_path)
    # Delete all files in directory_path
    for filename in os.listdir(directory_path):
        file = os.path.join(directory_path, filename)
        if os.path.isfile(file):
            os.remove(file)



def apply_order_by(data_chunk, column_name, sort_order):
    temp_folder = './data/temp_sorted'
    os.makedirs(temp_folder, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Check if the column contains numeric values
    is_numeric = pd.to_numeric(data_chunk[column_name], errors='coerce').notnull().all()
    
    if is_numeric:
        data_chunk[column_name] = pd.to_numeric(data_chunk[column_name], errors='coerce')
    
    if sort_order == 'STRAIGHT':
        data_chunk.sort_values(by=column_name, ascending=True, inplace=True, na_position='last')
      
    elif sort_order == 'REVERSE':
        data_chunk.sort_values(by=column_name, ascending=False, inplace=True, na_position='last')
    
    # Create a unique filename for each sorted chunk
    chunk_number = len(os.listdir(temp_folder)) + 1
    sorted_filename = f"sorted_data_chunk_{chunk_number}.csv"
    output_path = os.path.join(temp_folder, sorted_filename)
    
    data_chunk.to_csv(output_path, index=False)
    
    return output_path





def apply_single_condition(df, column, operator, value):
    if type(value)==str:
        value = value.strip("'")
    if operator == '=':
        return df[df[column] == value]
    elif operator == '!=':
        return df[df[column] != value]
    elif operator == '>=':
        return df[df[column] >= value]
    elif operator == '<=':
        return df[df[column] <= value]
    elif operator == '<':
        return df[df[column] < value]
    elif operator == '>':
        return df[df[column] > value]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def store_chunk(data_chunk, file_path, file_name):
    chunk_number = len(os.listdir(file_path)) + 1
    sorted_filename = file_name + f"_{chunk_number}.csv"
    output_path = os.path.join(file_path, sorted_filename)
    data_chunk.to_csv(output_path, index=False)
    
import os
import csv

def combine_csv_files(folder_path, output_file):
    unique_rows = set()
    processed_files = []
    
    # Check if the output file exists, if yes, load its unique rows
    if os.path.exists(output_file):
        df_output = pd.read_csv(output_file)
        unique_rows = set(map(tuple, df_output.values.tolist()))
    
    file_list = os.listdir(folder_path)
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv') and file_name != output_file:
            df = pd.read_csv(file_path)
            for row in df.values.tolist():
                row_tuple = tuple(row)
                if row_tuple not in unique_rows:
                    unique_rows.add(row_tuple)
            processed_files.append(file_path)
    
    # Write unique rows to the output CSV
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Convert the header to a list and write it
        writer.writerow(list(df.columns))
        # Convert each tuple to a list and write the rows
        writer.writerows(map(list, unique_rows))
    


def apply_conditions(df, conditions,condition_number):
    filter_path = "./data/temp_filter"
    output_file = "./data/temp_filter/condition_result" +condition_number + '.csv'
    if not conditions:
        return df
    
    operator, column, value = conditions

    filtered_df = apply_single_condition(df, column, operator, value) 

    #store_chunk(filtered_df, filter_path, "filtered_data_chunk")
    if os.path.exists(output_file):
        # Append the filtered data to the existing file
        filtered_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # Write the filtered data to a new file
        filtered_df.to_csv(output_file, index=False)

    return filtered_df


def delete_all_files(directory_path):
    """
    Delete all files within the specified directory.

    Args:
    - directory_path (str): Path to the directory containing files to be deleted.
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def row_exists_in_file(row, filename):
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() == row:
                return True
    return False

# Function to append unique rows from file2 to file1
def append_unique_rows(file1_path, file2_path):
    with open(file1_path, 'a') as file1:
        with open(file2_path, 'r') as file2:
            for line in file2:
                row = line.strip()
                if not row_exists_in_file(row, file1_path):
                    file1.write(row + '\n')


def get_csv_files_in_folder(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    return csv_files

def read_csv_in_chunks(file_path, chunk_size=1000):
    with open(file_path, 'r') as file:
        header = file.readline().strip().split(',')  # Read and split the header line
        index_counter = 0  # Initialize index counter
        while True:
            chunk = []  # Initialize an empty list for the chunk
            for _ in range(chunk_size):
                line = file.readline()
                if not line:
                    break  # Break if end of file is reached
                chunk.append(line.strip().split(','))  # Append row data
                index_counter += 1  # Increment index counter
            if len(chunk) > 0:  # Check if there's data in the chunk
                chunk = [[index_counter - len(chunk) + i] + row for i, row in enumerate(chunk)]
                # Create DataFrame with 'index' column
                df_chunk = pd.DataFrame(chunk, columns=['index'] + header)
                df_chunk.set_index('index', inplace=True)  # Set 'index' column as the index
                
                for col in df_chunk.columns:
                    try:
                        # Attempt to convert to numeric
                        df_chunk[col] = pd.to_numeric(df_chunk[col])
                    except ValueError:
                        # If conversion fails, keep the column as object type
                        df_chunk[col] = df_chunk[col].astype(object)
                
                yield df_chunk
            else:
                break




def store_chunk(data_chunk, file_path, file_name):
    chunk_number = len(os.listdir(file_path)) + 1
    sorted_filename = file_name + f"_{chunk_number}.csv"
    output_path = os.path.join(file_path, sorted_filename)
    data_chunk.to_csv(output_path, index=False)

def store_file(data, file_path, file_name):
    sorted_filename = file_name + ".csv"
    output_path = os.path.join(file_path, sorted_filename)
    data.to_csv(output_path, index=False)
    

def combine_csv_files(folder_path, output_file):
    unique_rows = set()
    processed_files = []
    
    # Check if the output file exists, if yes, load its unique rows
    if os.path.exists(output_file):
        df_output = pd.read_csv(output_file)
        unique_rows = set(map(tuple, df_output.values.tolist()))
    
    file_list = os.listdir(folder_path)
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.csv') and file_name != output_file:
            df = pd.read_csv(file_path)
            for row in df.values.tolist():
                row_tuple = tuple(row)
                if row_tuple not in unique_rows:
                    unique_rows.add(row_tuple)
            processed_files.append(file_path)
    
    # Write unique rows to the output CSV
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Convert the header to a list and write it
        writer.writerow(list(df.columns))
        # Convert each tuple to a list and write the rows
        writer.writerows(map(list, unique_rows))
    




def delete_all_files(directory_path):
    """
    Delete all files within the specified directory.

    Args:
    - directory_path (str): Path to the directory containing files to be deleted.
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def row_exists_in_file(row, filename):
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() == row:
                return True
    return False

# Function to append unique rows from file2 to file1
def append_unique_rows(file1_path, file2_path):
    with open(file1_path, 'a') as file1:
        with open(file2_path, 'r') as file2:
            for line in file2:
                row = line.strip()
                if not row_exists_in_file(row, file1_path):
                    file1.write(row + '\n')

def dict_to_dataframe(data_dict, group_by_column):
    if group_by_column == None:
        df = pd.DataFrame(data_dict,[1])
        df = df.transpose()
        return df
    else:
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        return df.dropna()


def custom_aggregation( agg_func, group_by_column=None, aggregate_column=None):
    aggregate_dict = {}
    current_group_value = None
    current_group_data = pd.DataFrame()
    files = get_csv_files_in_folder("./data/result")
    file_path = "./data/result/" + files[0]

    flag = True
    for chunk in read_csv_in_chunks(file_path, chunk_size=1000):
        for index, row in chunk.iterrows():
            if group_by_column:
                group_value = row[group_by_column]
                if group_value != current_group_value:
                    if current_group_value is not None:
                        aggregated_value = agg_func(current_group_data, aggregate_column)
                        aggregate_dict[current_group_value] = {
                            f"{aggregate_column}_{agg_func.__name__}": aggregated_value
                        }
                    current_group_value = group_value
                    current_group_data = pd.DataFrame([row])
                else:

                    new_index = len(current_group_data)
                    current_group_data.loc[new_index] = row
                    #current_group_data = pd.concat([current_group_data, row], ignore_index=True)
                    #print(current_group_data)
            else:
                if flag:
                    current_group_data = pd.DataFrame([row])
                    flag =False
                new_index = len(current_group_data)
                current_group_data.loc[new_index] = row
                #current_group_data = pd.concat([current_group_data, row], ignore_index=True)
 
    if group_by_column and not current_group_data.empty and current_group_value is not None:
        aggregated_value = agg_func(current_group_data, aggregate_column)
        aggregate_dict[current_group_value] = {
            f"{aggregate_column}_{agg_func.__name__}": aggregated_value
        }

    elif not group_by_column and not current_group_data.empty:  # Aggregate across all data chunks
        aggregated_value = agg_func(current_group_data, aggregate_column)
        aggregate_dict[f"Total {aggregate_column}_{agg_func.__name__}"] = aggregated_value
    

    aggregate_df = dict_to_dataframe(aggregate_dict, group_by_column)

    return aggregate_df




def custom_sum(data, column_name):
    return sum(row[column_name] for index, row in data.iterrows())

def custom_min(data, column_name):
    min_val = float('inf')
    for index, row in data.iterrows():
        val = row[column_name]
        if val < min_val:
            min_val = val
    return min_val

def custom_max(data, column_name):
    max_val = float('-inf')
    for index, row in data.iterrows():
        val = row[column_name]
        if val > max_val:
            max_val = val
    return max_val

def custom_average(data, column_name):
    total_sum = 0
    count = 0
    for index, row in data.iterrows():
        total_sum += row[column_name]
        count += 1
    return total_sum / count if count != 0 else 0  # Avoid division by zero

def custom_count(data, column_name):
    count = 0
    for index, row in data.iterrows():
        count += 1
    return count

def aggregate_handler(aggregate_functions,group_by_column):
        check1 = list(aggregate_functions)
        if type(check1[0]) == tuple:
            pass
        else:
            aggregate_functions= [tuple(aggregate_functions)]
        
        result = [] # initialize result list of aggregate columns
        for func, col in aggregate_functions:
 
    
            if func  == 'SUM':
                result.append(custom_aggregation(custom_sum, group_by_column, col)) 
            elif func  == 'LEAST':
                result.append(custom_aggregation(custom_min, group_by_column, col))
            elif func  == 'MOST':
                result.append(custom_aggregation(custom_max, group_by_column, col))
            elif func  == 'MEAN':
                result.append(custom_aggregation(custom_average, group_by_column, col))
            elif func  == 'COUNT':
                result.append(custom_aggregation(custom_count, group_by_column, col))
            else:
                print('error')
                
            #results[group][f'{func}_{col}'] = apply_aggregate_function((func, col), group_df)
        result_list = result
        if group_by_column != None:
            merged_df = pd.concat(result_list,axis = 1)
            merged_df = merged_df.rename_axis(group_by_column)
        else:
            merged_df = pd.concat(result_list,axis = 1)
            merged_df = merged_df.T
            for column in merged_df.columns:
                non_nan_values = merged_df[column].dropna()
                if not non_nan_values.empty:
                    merged_df.at[0, column] = non_nan_values.values[0]
            merged_df = merged_df.iloc[[-1]]
            merged_df = merged_df.rename_axis('Index')
        return merged_df

group_names = []

def perform_operation(operation, parameter,group_by_column =None):
    global group_names
    file_path_list = []
    result_files = get_csv_files_in_folder('./data/result')
    file_path = './data/result/'  + result_files[0]
    
    unique_values =[]
    # Read data in chunks of 20 rows
    for data_chunk in read_csv_in_chunks(file_path, chunk_size=1000):
        if operation == "projection":
            apply_projection(data_chunk, parameter)
        elif operation == "orderby":
            column_name, sort_order = parameter
            sorted_chunk_file_path = apply_order_by(data_chunk, column_name, sort_order)
        elif operation == "filter":
            cond_number = ''
            # Applying conditions sequentially and saving intermediate results
            filter_path = "./data/temp_filter"
            if parameter[0] != 'AND' and parameter[0] != 'OR': 
                apply_conditions(data_chunk, parameter,'')
            else:
                AND_OR = parameter[0]
             
                cond1 = parameter[1]
                cond2 = parameter[2]
                df = apply_conditions(data_chunk, cond1,'')
                
                if parameter[0] == 'AND':
                  
                    apply_conditions(df, cond2,'2')
                    cond_number = '2'
                elif parameter[0] == 'OR':
                    apply_conditions(data_chunk, cond2,'2')
                    condition_result_file = './data/temp_filter/condition_result.csv'
                    condition_result2_file = './data/temp_filter/condition_result2.csv'
                    append_unique_rows(condition_result_file, condition_result2_file)
        
        elif operation == "groupby":
            specified_column = parameter[0]
            for index, row in data_chunk.iterrows():
                value = row[specified_column]
                if value not in unique_values:
                    unique_values.append(value)


    if operation == "orderby":
        sorted_chunk_file_path = merge_and_cleanup_files( column_name, sort_order)
    if operation == "filter":
        
        result_file_path = './data/temp_filter/condition_result'+ cond_number+ '.csv'
        
        result_folder_path = './data/result'
        # Delete all files in result folder
        
        delete_all_files(result_folder_path)
        # Create result file 
        
        shutil.move(result_file_path, os.path.join(result_folder_path, 'result.csv'))
        delete_all_files(filter_path)
    if operation == "groupby":
        group_names = unique_values
        for unique_value in unique_values:
            # Loop over each unique value
            for data_chunk in read_csv_in_chunks(file_path, chunk_size=1000):
                # Check if groupby_result.csv exists in the folder
                if not os.path.exists("./data/temp_groupby/groupby_result.csv"):
                    # Filter rows with the same unique value
                    filtered_rows = data_chunk[data_chunk[specified_column] == unique_value]
                
                    # Save the filtered rows to groupby_result.csv
                    filtered_rows.to_csv("./data/temp_groupby/groupby_result.csv", index=False)
                else:
                    # Append rows to the existing groupby_result.csv
                    filtered_rows = data_chunk[data_chunk[specified_column] == unique_value]
                   
                    filtered_rows.to_csv("./data/temp_groupby/groupby_result.csv", mode='a', index=False, header=False)
        result_file_path = './data/temp_groupby/groupby_result.csv'
        
        result_folder_path = './data/result'
        # Delete all files in result folder
        
        delete_all_files(result_folder_path)
        # Create result file 
        
        shutil.move(result_file_path, os.path.join(result_folder_path, 'result.csv'))
        delete_all_files('./data/temp_groupby')
    if operation == 'aggregate':
        result_folder_path = './data/result'
        aggregate_functions = parameter
        if group_by_column != None:
            group_by_column = group_by_column[0]
        df= aggregate_handler(aggregate_functions,group_by_column)
        if group_by_column!= None:
            df[group_by_column] = group_names
        delete_all_files(result_folder_path)
        store_file(df,result_folder_path, 'result')
        column_list = df.columns.tolist()
        return column_list
            

# Define the parsing rules
def p_statement(p):
    '''
    statement : select_statement_with_where
              | select_statement_with_order_by
              | select_statement_with_aggregate
              | select_aggregate_without_where_group_by
              | select_aggregate_with_where_group_by
              | select_statement_group_by
              
    ''' 
    p[0] = p[1]

def p_update_statement(p):
    '''
    update_statement : UPDATE table_list SET set_list where_clause
    '''
    table_name = p[2]
    set_values = p[4]
    conditions = p[5]

    # Your code here to perform the update operation
    #update_query(table_name, set_values, conditions)


def p_set_list(p):
    '''
    set_list : set_expression
             | set_list COMMA set_expression
    '''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_set_expression(p):
    '''
    set_expression : IDENTIFIER EQUALS expression
    '''
    p[0] = (p[1], p[3])  # Store column name and new value in a tuple



def p_where_clause(p):
    '''
    where_clause : MENTIONED condition
    '''
    p[0] = p[2]
   

def p_condition(p):
    '''
    condition : expression
              | expression AND condition
              | expression OR condition
    '''
  
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = (p[2], p[1], p[3])


def p_expression(p):
    '''
    expression : IDENTIFIER EQUALS expression
               | IDENTIFIER NOT_EQUALS expression
               | IDENTIFIER LT expression
               | IDENTIFIER GT expression
               | NUMBER
               | STRING
    '''
    # Your code here to handle expressions
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = (p[2], p[1], p[3])

def p_order_by_clause(p):
    '''
    order_by_clause : SORT IDENTIFIER STRAIGHT
                    | SORT IDENTIFIER REVERSE
    '''

    p[0] = (p[2], p[3])  # Store the column and sorting order (STRAIGHT or REVERSE)


def p_aggregate_function(p):
    '''
    aggregate_function : COUNT LPAREN IDENTIFIER RPAREN
                       | SUM LPAREN IDENTIFIER RPAREN
                       | MEAN LPAREN IDENTIFIER RPAREN
                       | MOST LPAREN IDENTIFIER RPAREN
                       | LEAST LPAREN IDENTIFIER RPAREN
    '''
    p[0] = (p[1], p[3]) 

def p_group_by_clause(p):
    '''
    group_by_clause : CLUSTER column_list
    '''
    p[0] = p[2]



def p_select_statement_with_where(p):
    '''
    select_statement_with_where : PRESENT column_list THIS table_list where_clause order_by_clause
                                | PRESENT column_list THIS table_list where_clause
    '''
    selected_columns = p[2]
    tables = p[4] if len(p) > 4 else None
    conditions = p[5] if len(p) > 5 else None
    order_by = p[6] if len(p) > 6 else None

    
    #columns = get_all_column_names(tables)
    #result = df[columns].copy()
    if conditions:

        perform_operation('filter',conditions)

        #result = apply_conditions(result, conditions)

    if order_by:
        pass
        perform_operation('orderby',order_by)
        #result = apply_order_by(result, order_by)
    
    #print(result[selected_columns], selected_columns)
    print('Final Result')
    perform_operation('projection',selected_columns)
    # Your code here, use columns and tables accordingly

def p_select_statement_group_by(p):
    '''
    select_statement_group_by : PRESENT column_list THIS table_list group_by_clause
    '''
    selected_columns = p[2]
    tables = p[4] if len(p) > 4 else None
    group_by = p[5] if len(p) > 5 else None

    
    #columns = get_all_column_names(tables)
    #result = df[columns].copy()

    if group_by:
        perform_operation('groupby',group_by)
        #result = apply_group_by(result, group_by)

    
    #print(result[selected_columns], selected_columns)
    print('Final Result')
    perform_operation('projection',selected_columns)
    # Your code here, use columns and tables accordingly

def p_select_statement_with_order_by(p):
    '''
    select_statement_with_order_by : PRESENT column_list THIS table_list order_by_clause
                                   | PRESENT column_list THIS table_list
    '''
    selected_columns = p[2]
    tables = p[4] if len(p) > 4 else None
    order_by = p[5] if len(p) > 5 else None

    #columns = get_all_column_names(tables)

    if order_by:
        perform_operation("orderby",order_by)

    perform_operation("projection",selected_columns)

def p_select_statement_with_aggregate(p):
    '''
    select_statement_with_aggregate : PRESENT aggregate_function_list THIS table_list where_clause order_by_clause
                                    | PRESENT aggregate_function_list THIS table_list where_clause
                                    | PRESENT aggregate_function_list THIS table_list
    '''

    aggregate_function = p[2]
    tables = p[4] if len(p) > 4 else None
    conditions = p[5] if len(p) > 5 else None
    order_by = p[6] if len(p) > 6 else None

    #columns = get_all_column_names(tables)
    
    # Apply aggregate function to columns
    #result = df[columns].copy()
    
    if conditions:

        perform_operation('filter',conditions)

    if order_by:
        perform_operation('orderby',order_by)
        #result = apply_order_by(result, order_by)
    
    print("aggregate result")

    columns_list = perform_operation('aggregate',aggregate_function, None)
    perform_operation("projection",columns_list)


def p_select_aggregate_with_where_group_by(p):
    '''
    select_aggregate_with_where_group_by : PRESENT aggregate_function_list THIS table_list where_clause group_by_clause order_by_clause
                                         | PRESENT aggregate_function_list THIS table_list where_clause group_by_clause
    '''
    aggregate_function = p[2]
    tables = p[4]
    conditions = p[5]
    group_by = p[6]
    order_by = p[7] if len(p) > 7 else None

    # Apply the aggregate function with MENTIONED and CLUSTER
    #columns = get_all_column_names(tables)
    #result = df[columns].copy()

    if conditions:

        perform_operation('filter',conditions)

    if group_by:
        perform_operation('groupby',group_by)
        #result = apply_group_by(result, group_by)


    columns_list = perform_operation('aggregate',aggregate_function, group_by)

    if order_by:
        perform_operation('orderby',order_by)
        #result = apply_order_by(result, order_by)
    perform_operation("projection",columns_list)

    # Print or return the result as needed

def p_select_aggregate_without_where_group_by(p):
    '''
    select_aggregate_without_where_group_by : PRESENT aggregate_function_list THIS table_list group_by_clause order_by_clause
                                             | PRESENT aggregate_function_list THIS table_list group_by_clause
    '''
    aggregate_function = p[2]
    tables = p[4]
    group_by = p[5]
    order_by = p[6] if len(p) > 6 else None

    # Apply the aggregate function with only CLUSTER
    #columns = get_all_column_names(tables)
    #result = df[columns].copy()

    if group_by:
        perform_operation('groupby',group_by)
        #result = apply_group_by(result, group_by)

    if order_by:
        perform_operation('orderby',order_by)
        #result = apply_order_by(result, order_by)
    
    #result = apply_aggregate_functions(aggregate_function, result)
    columns_list = perform_operation('aggregate',aggregate_function, group_by)
    perform_operation("projection",columns_list)
    #print(result)
    # Print or return the result as needed


def p_aggregate_function_list(p):
    '''
    aggregate_function_list : aggregate_function
                            | aggregate_function_list COMMA aggregate_function
    '''
    if len(p) == 2:
        p[0] = [p[1]]  # Single aggregate function
    else:
        p[0] = p[1] + [p[3]]  # List of aggregate functions separated by commas

def p_column_list(p):
    '''
    column_list : ASTERISK
                | IDENTIFIER
                | column_list COMMA IDENTIFIER
    '''

    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 4:
        p[0] = p[1] + [p[3]]


    # Your code here, use p[1] as column name

def p_table_list(p):
    '''
    table_list : IDENTIFIER
               | table_list COMMA IDENTIFIER
    '''
    # Your code here, use p[1] as table name
    p[0] = p[1]

    source_folder = "./data/"
    destination_folder = "./data/result/"

    file_name = p[0]+'.csv'  # Replace with your file name

    # Copying the file from source to destination
    delete_all_files("./data/result")
    shutil.copyfile(source_folder + file_name, destination_folder + file_name)



def get_all_column_names(table_name):
    columns = list(df.columns)
    return columns




#svk

def initialize_parser():
    return yacc.yacc()




# Function to perform an CHANGE query
'''
def perform_update_query(parsed_query, folder_path):
    if parsed_query:
        table_name, column_name, new_value, condition_column, operator, condition_value = parsed_query
        
        # Constructing the file path
        file_path = os.path.join(folder_path, f"{table_name}.csv")
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the CSV file
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            
            # Update rows based on the condition
            for row in rows:
                if eval(f"{row[condition_column]} {operator} {condition_value}"):
                    row[column_name] = new_value
            
            # Write updated rows back to the CSV file
            with open(file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"Query performed successfully. Updated {column_name} in {table_name}.csv where {condition_column} {operator} {condition_value}")
        else:
            print(f"File {table_name}.csv does not exist in the specified folder.")
    else:
        print("Invalid parsed query.")

# Function to perform an ADD query
def perform_insert_query(parsed_query, folder_path):
    if parsed_query:
        table_name = parsed_query[0]
        columns = parsed_query[1]
        values = parsed_query[2]
        
        # Constructing the file path
        file_path = os.path.join(folder_path, f"{table_name}.csv")
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Insert rows into the CSV file
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(values)
            
            print(f"Query performed successfully. Inserted values into {table_name}.csv")
        else:
            print(f"File {table_name}.csv does not exist in the specified folder.")
    else:
        print("Invalid parsed query.")
'''

def perform_delete_query(parsed_query, folder_path):
    if parsed_query:
        table_name, condition_column, operator, condition_value = parsed_query
        
        # Constructing the file path
        file_path = os.path.join(folder_path, f"{table_name}.csv")
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the CSV file
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            
            # Filter rows based on the condition and create a new list without the matching rows
            filtered_rows = [row for row in rows if not eval(f"{row[condition_column]} {operator} {condition_value}")]
            
            # Write updated rows back to the CSV file (filtered rows)
            with open(file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)
            
            print(f"Query performed successfully. Deleted rows from {table_name}.csv where {condition_column} {operator} {condition_value}")
        else:
            print(f"File {table_name}.csv does not exist in the specified folder.")
    else:
        print("Invalid parsed query.")

# Function to parse SQL queries (CHANGE, ADD, EVAPORATE)
def parse_sql_query(query):
    update_pattern = r"CHANGE\s+(\w+)\s+FOR\s+(\w+)\s*=\s*(\S+)\s+MENTIONED\s+(\w+)\s*([=<>]+)\s*(\S+)"
    insert_pattern = r"ADD THIS\s+(\w+)\s+\((.*?)\)\s+VALUES\s+\((.*?)\)"
    delete_pattern = r"EVAPORATE THIS\s+(\w+)\s+MENTIONED\s+(\w+)\s*([=<>]+)\s*(\S+)"
    
    update_match = re.match(update_pattern, query)
    insert_match = re.match(insert_pattern, query)
    delete_match = re.match(delete_pattern, query)
    
    if update_match:
        table_name = update_match.group(1)
        column_name = update_match.group(2)
        new_value = update_match.group(3)
        condition_column = update_match.group(4)
        operator = update_match.group(5)
        condition_value = update_match.group(6)
        return ["CHANGE", table_name, column_name, new_value, condition_column, operator, condition_value]
    
    elif insert_match:
        table_name = insert_match.group(1)
        columns = [col.strip() for col in insert_match.group(2).split(',')]
        values = [val.strip() for val in insert_match.group(3).split(',')]
        return ["ADD", table_name, columns, values]
    elif delete_match:
        table_name = delete_match.group(1)
        condition_column = delete_match.group(2)
        operator = delete_match.group(3)
        condition_value = delete_match.group(4)
        
        return ["EVAPORATE", table_name, condition_column, operator, condition_value]
    else:
        return None



def read_csv_in_chunks(file_path, chunk_size=1000):
    with open(file_path, 'r') as file:
        header = file.readline().strip().split(',')  # Read and split the header line
        #print(header)
        index_counter = 0  # Initialize index counter
        while True:
            chunk = []  # Initialize an empty list for the chunk
            for _ in range(chunk_size):
                line = file.readline()
                if not line:
                    break  # Break if end of file is reached
                chunk.append(line.strip().split(','))  # Append row data
                index_counter += 1  # Increment index counter
            if len(chunk) > 0:  # Check if there's data in the chunk
                chunk = [[index_counter - len(chunk) + i] + row for i, row in enumerate(chunk)]
                # Create DataFrame with 'index' column
                df_chunk = pd.DataFrame(chunk, columns=['index'] + header)
                df_chunk.set_index('index', inplace=True)  # Set 'index' column as the index
                for col in df_chunk.columns:
                    try:
                        # Attempt to convert to numeric
                        df_chunk[col] = pd.to_numeric(df_chunk[col])
                    except ValueError:
                        # If conversion fails, keep the column as object type
                        df_chunk[col] = df_chunk[col].astype(object)
                
                yield df_chunk
            else:
                break

# Function to perform SQL UPDATE, INSERT AND DELETE queries on CSV in chunks
def perform_ddl_queries_on_chunks(file_path, buffer_folder_path, query, chunk_size=1000):
    os.makedirs(buffer_folder_path)
    last_chunk =None
    count = 0
    for data_chunk in read_csv_in_chunks(file_path, chunk_size):
        if query.startswith("CHANGE"):
            parsed_query = parse_sql_query(query)
            if parsed_query and parsed_query[0] == "CHANGE":
                perform_update_query_on_chunk(parsed_query[1:], data_chunk, buffer_folder_path)
        
        elif query.startswith("ADD"):
            count = count +1
            last_chunk = data_chunk
        
        elif query.startswith("EVAPORATE"):
            parsed_query = parse_sql_query(query)
            if parsed_query and parsed_query[0] == "EVAPORATE":
                perform_delete_query_on_chunk(parsed_query[1:], data_chunk, buffer_folder_path)
    
    if count>0:
        new_count = 0
        for data_chunk in read_csv_in_chunks(file_path, chunk_size):
            new_count = new_count +1 
            parsed_query = parse_sql_query(query)
            if new_count != count:
                perform_insert_query_on_chunk(parsed_query[1:], data_chunk, buffer_folder_path)
            else:
                parsed_query = parsed_query[1:]
                table_name = parsed_query[0]
                columns = parsed_query[1]
                values = parsed_query[2]
                new_row1 = pd.DataFrame([values], index=columns)
                new_index = len(data_chunk)
                data_chunk.loc[new_index] = values
                save_chunk_to_buffer(data_chunk, table_name, buffer_folder_path)
                
                
        
                
            
        
        
    # After loop completion, replace the original file with the file in the buffer folder
    replace_file_in_buffer(file_path, buffer_folder_path)

# Function to perform an CHANGE query on a chunk of data and save the modified chunk in a buffer folder
def perform_update_query_on_chunk(parsed_query, data_chunk, buffer_folder_path):
    if parsed_query and data_chunk is not None:
        table_name, column_name, new_value, condition_column, operator, condition_value = parsed_query
        
        # Update rows based on the condition
        data_chunk.loc[eval(f"data_chunk['{condition_column}'] {operator} {condition_value}"), column_name] = new_value
        
        # Save the modified chunk in the buffer folder
        save_chunk_to_buffer(data_chunk, table_name, buffer_folder_path)

# Function to perform an ADD query on a chunk of data and save the modified chunk in a buffer folder
def perform_insert_query_on_chunk(parsed_query, data_chunk, buffer_folder_path):
    if parsed_query and data_chunk is not None:
        table_name = parsed_query[0]
        columns = parsed_query[1]
        values = parsed_query[2]
        save_chunk_to_buffer(data_chunk, table_name, buffer_folder_path)
        
    

# Function to perform a EVAPORATE query on a chunk of data and save the modified chunk in a buffer folder
def perform_delete_query_on_chunk(parsed_query, data_chunk, buffer_folder_path):
    if parsed_query and data_chunk is not None:
        table_name, condition_column, operator, condition_value = parsed_query
        
        # Filter rows based on the condition and create a new chunk without the matching rows
        data_chunk = data_chunk.loc[~eval(f"data_chunk['{condition_column}'] {operator} {condition_value}")]
        
        # Save the modified chunk in the buffer folder
        save_chunk_to_buffer(data_chunk, table_name, buffer_folder_path)

# Function to save a chunk of data into a buffer folder as a CSV file
def save_chunk_to_buffer(data_chunk, table_name, buffer_folder_path):
    buffer_file_path = os.path.join(buffer_folder_path, f"{table_name}_buffer.csv")
    if os.path.exists(buffer_file_path):
        # Append data to existing file
        data_chunk.to_csv(buffer_file_path, mode='a', header=False, index=False)
    else:
        # Create new file and write data
        data_chunk.to_csv(buffer_file_path, index=False)

# Function to replace the original file with the file in the buffer folder
def replace_file_in_buffer(original_file_path, buffer_folder_path):
    for file_name in os.listdir(buffer_folder_path):
        if file_name.endswith("_buffer.csv"):
            buffer_file_path = os.path.join(buffer_folder_path, file_name)
            os.replace(buffer_file_path, original_file_path)
            break
    # Delete the buffer folder
    os.rmdir(buffer_folder_path)

# Example SQL query

def execute_user_queries():
    # Replace 'file_path' and 'buffer_folder_path' with your specific file and folder paths
    file_path = "./data/crime_dataset.csv"
    buffer_folder_path = "./data/buffer_folder"

    while True:
        # Taking input from the user for the SQL query
        user_query = input("Enter your SQL query (Type 'exit' to quit): ")

        if user_query.lower() == 'exit':
            print("Exiting...")
            break

        # Check if the first word of the query is CHANGE, ADD, or EVAPORATE
        first_word = user_query.split()[0].lower()
        if first_word in ['change', 'add', 'evaporate']:
            # Perform SQL queries on chunks using user input query
            perform_ddl_queries_on_chunks(file_path, buffer_folder_path, user_query)
        else:
            # Perform some other operation for queries that don't start with CHANGE, ADD, or EVAPORATE
            print("Performing operation for this query:", user_query)
            tokens = tokenize_sql_input(user_query)
            # Parse
            parser = initialize_parser()
            parsed_query = parser.parse(user_query)



# The parser should print the result of the query (subset of DataFrame columns)
def main():
    
    sql_input1 = "PRESENT Column4,Column2 THIS test_data" 
    # Where commands Working
    sql_input2 = "PRESENT Column1,Column2 THIS test_data MENTIONED Column2 = 10" 
    sql_input3 = "PRESENT Column1,Column2 THIS test_data MENTIONED Column2 = 10 AND Column2 = 33"
    sql_input4 = "PRESENT Column1,Column2 THIS test_data MENTIONED Column2 = 10 OR Column2 = 33"
    sql_input5 = "PRESENT Column3 THIS test_data MENTIONED Column3 = 'A'" 
    sql_input6 = "PRESENT Column2 THIS test_data MENTIONED Column3 = 'A'" 
    #Order_by commands
    sql_input7 = "PRESENT Column2 THIS test_data SORT Column2 STRAIGHT"
    sql_input8 = "PRESENT Column2, Column3 THIS test_data MENTIONED Column3 = 'A' SORT Column2 STRAIGHT"
    sql_input9 = "PRESENT Column2, Column3 THIS test_data  SORT Column3 REVERSE"
    sql_input10 = "PRESENT Column2 THIS test_data MENTIONED Column3 = 'A' SORT Column2 REVERSE"
    sql_input11 = "PRESENT Column1 THIS test_data SORT Column3 REVERSE"  
    sql_input12 = "PRESENT Column1 THIS test_data MENTIONED Column3 = 'B' SORT Column2 REVERSE" 
    #Aggregreate functions
    sql_input13 = "PRESENT LEAST(Column2), LEAST(Column1) THIS test_data"
    sql_input14 = "PRESENT SUM(Column2) THIS test_data"
    sql_input15 = "PRESENT SUM(Column2) THIS test_data MENTIONED Column3 = 'A'" 
    sql_input16 = "PRESENT LEAST(Column2) THIS test_data MENTIONED Column3 = 'A'"
    sql_input17 = "PRESENT COUNT(Column3) THIS test_data"


    # Group_By 
    sql_input18 = "PRESENT SUM(Column2) THIS test_data"
    sql_input19 = "PRESENT MEAN(Column1),MEAN(Column2) THIS test_data MENTIONED Column3 != 'A' CLUSTER Column3 SORT Column1_custom_average STRAIGHT"
    sql_input20 = "PRESENT Column2,Column3 THIS test_data CLUSTER Column4"

    sql_input21 = "UPDATE test_data SET grade = 3 MENTIONED subject = 5"


    cd_input1 = "PRESENT crime_month,crime_city THIS crime_dataset" 
    cd_input2 = "PRESENT crime_city,severity_level THIS crime_dataset MENTIONED severity_level = 7 OR severity_level = 5"
    cd_input3 = "PRESENT crime_city,current_status,victim_count THIS crime_dataset MENTIONED crime_city = 'Los Angeles' SORT victim_count STRAIGHT"
    cd_input4 = "PRESENT LEAST(victim_count),MEAN(severity_level) THIS crime_dataset MENTIONED crime_city = 'Los Angeles'"
    cd_input5 = "PRESENT AVG(victim_count) THIS crime_dataset CLUSTER crime_city"
    cd_input6 =  "PRESENT MEAN(victim_count),MEAN(severity_level) THIS crime_dataset MENTIONED crime_city = 'El Paso' CLUSTER crime_type SORT severity_level_custom_average STRAIGHT"
    
    execute_user_queries()

# Run the main function
if __name__ == "__main__":
    main()





