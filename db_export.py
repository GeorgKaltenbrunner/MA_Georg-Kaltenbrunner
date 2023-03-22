import psycopg2
import GlobaleVariables
import json


def export_to_db_features(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                          v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38,
                          v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56,
                          v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67):
    """

    Exports the feature data into the PostgreSQL DB.
    """
    with open('credentials.json') as cred:
        cred = json.load(cred)

    try:
        # Connect to database
        connection = psycopg2.connect(user=cred['user'],
                                      password=cred['password'],
                                      host=cred['host'],
                                      port=cred['port'],
                                      database=cred['database'])

        sql = """INSERT INTO MA_2023_features_test
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s)
        """
        # Create a cursor to perform database operations
        cursor = connection.cursor()
        cursor.execute(sql, (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                             v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38,
                             v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56,
                             v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67))

        # commit the changes to the database
        connection.commit()
        # close communication with the database
        cursor.close()
        connection.close()

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL 5", error)


def export_to_db_finished_orders(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12):
    """
    Exports the target values (sftt) into the target DB.
    """
    with open('credentials.json') as cred:
        cred = json.load(cred)

    try:
        # Connect to database
        connection = psycopg2.connect(user=cred['user'],
                                      password=cred['password'],
                                      host=cred['host'],
                                      port=cred['port'],
                                      database=cred['database'])

        sql = """INSERT INTO MA_2023_target_test
           VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           """
        # Create a cursor to perform database operations
        cursor = connection.cursor()
        cursor.execute(sql, (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12))

        # commit the changes to the database
        connection.commit()
        # close communication with the database
        cursor.close()
        connection.close()

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL 5", error)
