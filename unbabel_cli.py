import json
import pandas as pd
import argparse


def engine(options):
    """
    Loads the text file and builds a DataFrame with each json line.
    pandas does the heavy lifting working with the data and outputs
    it as a sequence (if applicable) of json lines with the results
    :param options: command line options
    """
    try:
        with open(options.input_file, 'r') as json_file:
            try:
                df = pd.DataFrame(json.loads(line) for line in json_file)
                if not set(['timestamp', 'duration', 'client_name', 'source_language', 'target_language']).issubset(
                       df.columns):
                    print("Can't find all the relevant fields to work with. Exiting...")
                    return
            except ValueError:
                print("That's not what I expected :(  (needs to be a text file with json lines). Exiting...")
                return
    except FileNotFoundError:
        print("Can't find that file. Exiting...")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.rename({'timestamp': 'date'}, axis=1, inplace=True)

    # Create a copy of dates for manipulation down the line which can't use the index
    # Index the DataFrame on dates and sort it
    df['date_temp'] = df['date']

    df.set_index('date', inplace=True)
    df.sort_values(by=['date'], inplace=True)

    # If a time window was specified let's drop all the entries out of scope. Let's keep the last entry before
    # the cutoff to have a real average on the first line of the final results.
    # To do this, a new column is added with the timestamp of the next record. This way, by comparing the desired
    # cutoff time with this new column it preserves the last record before that time.
    if options.window:
        df['next_timestamp'] = df['date_temp'].shift(-1)
        # Little hack to avoid having last entry removed. Since there's no next, keep self timestamp.
        df.at[df.index[-1], 'next_timestamp'] = df.at[df.index[-1], 'date_temp']

        cutoff_time = df.index[-1] - pd.Timedelta(minutes=options.window)
        df = df[df['next_timestamp'] > cutoff_time]

    # Apply additional filters. On argparse these have a regex wildcard default which now matches all records if
    # unchanged or attempts to match specified patterns.
    df = (df[
        (df['source_language'].str.contains(options.source)) &
        (df['target_language'].str.contains(options.target)) &
        (df['client_name'].str.contains(options.client))
        ])

    # To cover both instances of having missing datapoints in between a time window or multiple data points for a given
    # minute, two resamples are done. The first one squashes (mean) multiple records on a given minute and the
    # second one upscales for missing minutes with the previous mean. Drops irrelevant columns.
    df = df.loc[:, ['duration']]

    df = df.resample('1T', label='right').mean().dropna(how='all')
    df = df.resample('1T', closed='right').pad()

    # Once a lean DataFrame is achieved, the moving average is calculated. (min_periods=1 for initial moment)
    df['average_delivery_time'] = df['duration'].rolling(min_periods=1, window=2, center=True).mean()

    # Remove all but the moving average column (the date is the index at this stage) and keep only X records according
    # to specified time window.
    if options.window:
        df = df.loc[:, ['average_delivery_time']].tail(options.window)
    else:
        df = df.loc[:, ['average_delivery_time']]

    # Pandas can only pass date index as epoch or iso. To avoid further manipulation to match desired output, a
    # numerical index is restored and the date will be treated as a string. Entries are output as one json per line
    df = df.reset_index()
    df.update(df.loc[:, df.dtypes.astype(str).str.contains('date')].astype(str))

    print(df.to_json(orient='records', lines=True))


def setup():
    """
    captures command line arguments. optionals store a wildcard regex for possible search on engine()
    :return: parsed arguments with defaults for optionals left blank
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--input_file',
                        metavar='<filename>',
                        required=True,
                        help='text file with json lines')

    parser.add_argument('-w',
                        '--window',
                        '--window_size',
                        type=int,
                        metavar='<minutes>',
                        help=('Time period (minutes) to calculate the moving average. Example: 10 -- '
                              'If the most recent entry is from 17:11, it will be calculated '
                              'for every minute between 17:02 and 17:11. If not specified, range will be between first'
                              'and last entry.'))

    parser.add_argument('-c',
                        '--client',
                        type=str,
                        default='.*',
                        metavar='<name>',
                        help='Filter for a specific client.')

    parser.add_argument('-s',
                        '--source',
                        type=str,
                        default='.*',
                        metavar='<language>',
                        help=('Source language (en,fr,de,etc.) calculate only for translations with source text in this'
                              ' language'))

    parser.add_argument('-t',
                        '--target',
                        type=str,
                        default='.*',
                        metavar='<language>',
                        help='Target language (en,fr,de,etc.) calculate only when translation was for this language')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cli_options = setup()
    engine(cli_options)
