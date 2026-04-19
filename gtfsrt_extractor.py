"""
GTFS-RT Data Extractor for Scotland Central Belt Bus Delay Project
==================================================================

This script processes GTFS-RT protobuf files from the Data Library Archive
(data.datalibrary.uk) and extracts bus delay data for Scotland's Central Belt.

BEFORE RUNNING THIS SCRIPT:
1. Install dependencies:
   pip install gtfs-realtime-bindings protobuf pandas

2. Download ONE day of GTFS-RT data from:
   https://data.datalibrary.uk/transport/BODS-ARCHIVE/gtfsrt/
   Example: gtfsrt-20251001.zip (~5 GB)

3. Also download the matching timetable GTFS from:
   https://data.datalibrary.uk/transport/BODS-ARCHIVE/timetables/
   (This gives you stop_times.txt, trips.txt, routes.txt, agency.txt, stops.txt)

4. Extract the GTFS-RT zip into a folder (e.g., ./gtfsrt_20251001/)
   Extract the timetable GTFS into a folder (e.g., ./timetable/)

Usage:
   python gtfsrt_extractor.py --gtfsrt_dir ./gtfsrt_20251001/ --timetable_dir ./timetable/ --output scotland_delays.csv
"""

import os
import sys
import glob
import argparse
import zipfile
from datetime import datetime, timezone
from collections import defaultdict

try:
    from google.transit import gtfs_realtime_pb2
except ImportError:
    print("ERROR: Please install gtfs-realtime-bindings:")
    print("  pip install gtfs-realtime-bindings protobuf")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: Please install pandas:")
    print("  pip install pandas")
    sys.exit(1)


# ============================================================
# SCOTLAND CENTRAL BELT STOP PREFIXES
# ============================================================
# NaPTAN ATCO codes starting with these prefixes are in Scotland's Central Belt
# 600 = Glasgow area, 610 = Lanarkshire, 620 = Edinburgh/Lothians,
# 630 = Fife, 640 = Stirling/Falkirk/Clackmannanshire, 650 = West Dunbartonshire/East Renfrewshire
SCOTLAND_CENTRAL_BELT_PREFIXES = [
    '600',  # Glasgow City
    '601',  # East Dunbartonshire
    '602',  # East Renfrewshire
    '603',  # Inverclyde
    '604',  # Renfrewshire
    '605',  # West Dunbartonshire
    '606',  # North Lanarkshire
    '607',  # South Lanarkshire
    '608',  # East Ayrshire (border)
    '609',  # North Ayrshire (border)
    '610',  # South Ayrshire (border)
    '612',  # Argyll & Bute (partial)
    '620',  # City of Edinburgh
    '621',  # East Lothian
    '622',  # Midlothian
    '623',  # West Lothian
    '624',  # Scottish Borders (partial)
    '630',  # Fife
    '640',  # Clackmannanshire
    '641',  # Falkirk
    '642',  # Stirling
    '649',  # Perth & Kinross (partial)
]


def is_scotland_central_belt_stop(stop_id: str) -> bool:
    """Check if a stop_id belongs to Scotland's Central Belt based on NaPTAN ATCO prefix."""
    if not stop_id:
        return False
    for prefix in SCOTLAND_CENTRAL_BELT_PREFIXES:
        if stop_id.startswith(prefix):
            return True
    return False


def inspect_single_file(filepath: str, max_entities: int = 5):
    """
    Open ONE GTFS-RT protobuf file and print its raw structure.
    Use this to understand exactly what fields are available.
    """
    feed = gtfs_realtime_pb2.FeedMessage()

    # Handle both raw .pb files and nested .zip files
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            names = zf.namelist()
            if not names:
                print(f"Empty zip: {filepath}")
                return
            data = zf.read(names[0])
    else:
        with open(filepath, 'rb') as f:
            data = f.read()

    feed.ParseFromString(data)

    print("=" * 70)
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"GTFS-RT Version: {feed.header.gtfs_realtime_version}")
    print(f"Timestamp: {datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc)}")
    print(f"Total entities: {len(feed.entity)}")
    print("=" * 70)

    # Count entity types
    trip_updates = 0
    vehicle_positions = 0
    alerts = 0

    for entity in feed.entity:
        if entity.HasField('trip_update'):
            trip_updates += 1
        if entity.HasField('vehicle'):
            vehicle_positions += 1
        if entity.HasField('alert'):
            alerts += 1

    print(f"\nEntity breakdown:")
    print(f"  TripUpdates:      {trip_updates}")
    print(f"  VehiclePositions: {vehicle_positions}")
    print(f"  Alerts:           {alerts}")

    # Show sample TripUpdate entities
    print(f"\n{'='*70}")
    print(f"SAMPLE TRIP UPDATES (first {max_entities}):")
    print(f"{'='*70}")

    count = 0
    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue
        if count >= max_entities:
            break

        tu = entity.trip_update
        trip = tu.trip

        print(f"\n--- Entity ID: {entity.id} ---")
        print(f"  trip.trip_id:      {trip.trip_id}")
        print(f"  trip.route_id:     {trip.route_id}")
        print(f"  trip.direction_id: {trip.direction_id}")
        print(f"  trip.start_date:   {trip.start_date}")
        print(f"  trip.start_time:   {trip.start_time}")
        print(f"  trip.schedule_relationship: {trip.schedule_relationship}")

        if tu.HasField('vehicle'):
            print(f"  vehicle.id:    {tu.vehicle.id}")
            print(f"  vehicle.label: {tu.vehicle.label}")

        if tu.HasField('timestamp'):
            print(f"  timestamp: {datetime.fromtimestamp(tu.timestamp, tz=timezone.utc)}")

        print(f"  stop_time_updates: {len(tu.stop_time_update)}")

        for i, stu in enumerate(tu.stop_time_update[:3]):  # Show first 3 stops
            print(f"\n    Stop #{i+1}:")
            print(f"      stop_id:       {stu.stop_id}")
            print(f"      stop_sequence: {stu.stop_sequence}")

            if stu.HasField('arrival'):
                print(f"      arrival.delay: {stu.arrival.delay} seconds ({stu.arrival.delay/60:.1f} min)")
                if stu.arrival.time > 0:
                    print(f"      arrival.time:  {datetime.fromtimestamp(stu.arrival.time, tz=timezone.utc)}")
            else:
                print(f"      arrival: NOT PROVIDED")

            if stu.HasField('departure'):
                print(f"      departure.delay: {stu.departure.delay} seconds ({stu.departure.delay/60:.1f} min)")
                if stu.departure.time > 0:
                    print(f"      departure.time:  {datetime.fromtimestamp(stu.departure.time, tz=timezone.utc)}")
            else:
                print(f"      departure: NOT PROVIDED")

            print(f"      schedule_relationship: {stu.schedule_relationship}")

        if len(tu.stop_time_update) > 3:
            print(f"    ... and {len(tu.stop_time_update) - 3} more stops")

        count += 1

    # Show sample VehiclePosition entities
    print(f"\n{'='*70}")
    print(f"SAMPLE VEHICLE POSITIONS (first {max_entities}):")
    print(f"{'='*70}")

    count = 0
    for entity in feed.entity:
        if not entity.HasField('vehicle'):
            continue
        if count >= max_entities:
            break

        vp = entity.vehicle

        print(f"\n--- Entity ID: {entity.id} ---")
        if vp.HasField('trip'):
            print(f"  trip.trip_id:    {vp.trip.trip_id}")
            print(f"  trip.route_id:   {vp.trip.route_id}")
        if vp.HasField('vehicle'):
            print(f"  vehicle.id:      {vp.vehicle.id}")
            print(f"  vehicle.label:   {vp.vehicle.label}")
        if vp.HasField('position'):
            print(f"  position.lat:    {vp.position.latitude}")
            print(f"  position.lon:    {vp.position.longitude}")
            print(f"  position.bearing: {vp.position.bearing}")
            print(f"  position.speed:  {vp.position.speed}")
        print(f"  current_stop_sequence: {vp.current_stop_sequence}")
        print(f"  stop_id:               {vp.stop_id}")
        print(f"  current_status:        {vp.current_status}")
        if vp.timestamp > 0:
            print(f"  timestamp:             {datetime.fromtimestamp(vp.timestamp, tz=timezone.utc)}")

        count += 1

    # Show Scotland Central Belt stats
    print(f"\n{'='*70}")
    print(f"SCOTLAND CENTRAL BELT FILTER CHECK:")
    print(f"{'='*70}")

    scotland_trips = 0
    scotland_stops_found = set()
    total_stop_updates = 0

    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue
        is_scotland = False
        for stu in entity.trip_update.stop_time_update:
            total_stop_updates += 1
            if is_scotland_central_belt_stop(stu.stop_id):
                is_scotland = True
                scotland_stops_found.add(stu.stop_id[:3])
        if is_scotland:
            scotland_trips += 1

    print(f"  Total TripUpdates:            {trip_updates}")
    print(f"  Scotland Central Belt trips:  {scotland_trips}")
    print(f"  Total stop_time_updates:      {total_stop_updates}")
    print(f"  ATCO prefixes found:          {sorted(scotland_stops_found)}")
    print(f"  Scotland %:                   {scotland_trips/max(trip_updates,1)*100:.1f}%")


def extract_delays_from_file(filepath: str) -> list:
    """Extract delay records from a single GTFS-RT file, filtered for Scotland Central Belt."""
    feed = gtfs_realtime_pb2.FeedMessage()

    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            names = zf.namelist()
            if not names:
                return []
            data = zf.read(names[0])
    else:
        with open(filepath, 'rb') as f:
            data = f.read()

    try:
        feed.ParseFromString(data)
    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return []

    records = []
    feed_timestamp = feed.header.timestamp

    for entity in feed.entity:
        if not entity.HasField('trip_update'):
            continue

        tu = entity.trip_update
        trip = tu.trip

        for stu in tu.stop_time_update:
            # Filter: only Scotland Central Belt stops
            if not is_scotland_central_belt_stop(stu.stop_id):
                continue

            # Only include if arrival delay data exists
            if not stu.HasField('arrival'):
                continue

            record = {
                'feed_timestamp': datetime.fromtimestamp(feed_timestamp, tz=timezone.utc).isoformat(),
                'trip_id': trip.trip_id,
                'route_id': trip.route_id,
                'direction_id': trip.direction_id,
                'start_date': trip.start_date,
                'start_time': trip.start_time,
                'vehicle_id': tu.vehicle.id if tu.HasField('vehicle') else '',
                'stop_id': stu.stop_id,
                'stop_sequence': stu.stop_sequence,
                'arrival_delay_seconds': stu.arrival.delay,
                'arrival_delay_minutes': round(stu.arrival.delay / 60, 2),
                'arrival_time_unix': stu.arrival.time if stu.arrival.time > 0 else None,
                'departure_delay_seconds': stu.departure.delay if stu.HasField('departure') else None,
                'schedule_relationship': stu.schedule_relationship,
            }
            records.append(record)

    return records


def process_day(gtfsrt_dir: str, output_csv: str, sample_every_n: int = 60):
    """
    Process one day of GTFS-RT data.

    Since there are ~2880 files per day (one every 30 seconds), we sample
    every Nth file to avoid massive duplication. Default: every 60th file
    = roughly one sample every 30 minutes.

    Args:
        gtfsrt_dir: Directory containing the extracted day's .pb or .zip files
        output_csv: Path to output CSV
        sample_every_n: Process every Nth file (60 = ~30 min intervals)
    """
    # Find all protobuf files
    files = sorted(glob.glob(os.path.join(gtfsrt_dir, '*.pb')) +
                   glob.glob(os.path.join(gtfsrt_dir, '*.zip')) +
                   glob.glob(os.path.join(gtfsrt_dir, '**/*.pb'), recursive=True))

    if not files:
        print(f"No .pb or .zip files found in {gtfsrt_dir}")
        print("Make sure you extracted the daily zip file first.")
        return

    print(f"Found {len(files)} GTFS-RT files")
    print(f"Sampling every {sample_every_n}th file ({len(files)//sample_every_n} files to process)")

    all_records = []
    files_to_process = files[::sample_every_n]

    for i, filepath in enumerate(files_to_process):
        if i % 10 == 0:
            print(f"  Processing file {i+1}/{len(files_to_process)}: {os.path.basename(filepath)}")

        records = extract_delays_from_file(filepath)
        all_records.extend(records)

    if not all_records:
        print("\nNo Scotland Central Belt delay records found!")
        print("This might mean:")
        print("  1. The stop_id format in this feed doesn't use NaPTAN ATCO codes")
        print("  2. Scotland operators aren't in this particular snapshot")
        print("  3. The files need a different parsing approach")
        print("\nTry running: python gtfsrt_extractor.py --inspect <path_to_one_file>")
        return

    df = pd.DataFrame(all_records)

    # Deduplicate: keep latest feed_timestamp per trip+stop combination
    df = df.sort_values('feed_timestamp').drop_duplicates(
        subset=['trip_id', 'stop_id', 'start_date'], keep='last'
    )

    df.to_csv(output_csv, index=False)
    print(f"\n{'='*70}")
    print(f"SUCCESS! Extracted {len(df)} delay records")
    print(f"Saved to: {output_csv}")
    print(f"{'='*70}")
    print(f"\nColumns in your CSV:")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nDelay statistics (minutes):")
    print(df['arrival_delay_minutes'].describe())
    print(f"\nUnique routes: {df['route_id'].nunique()}")
    print(f"Unique stops:  {df['stop_id'].nunique()}")
    print(f"Unique trips:  {df['trip_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description='Extract Scotland bus delay data from GTFS-RT')

    parser.add_argument('--inspect', type=str,
                        help='Inspect a single GTFS-RT file to see its structure')
    parser.add_argument('--gtfsrt_dir', type=str,
                        help='Directory with extracted GTFS-RT .pb files for one day')
    parser.add_argument('--output', type=str, default='scotland_delays.csv',
                        help='Output CSV filename')
    parser.add_argument('--sample_every', type=int, default=60,
                        help='Sample every Nth file (default: 60 = ~30 min intervals)')

    args = parser.parse_args()

    if args.inspect:
        inspect_single_file(args.inspect)
    elif args.gtfsrt_dir:
        process_day(args.gtfsrt_dir, args.output, args.sample_every)
    else:
        parser.print_help()
        print("\n\nQUICK START:")
        print("=" * 50)
        print("\nStep 1: Inspect one file to see what's inside:")
        print("  python gtfsrt_extractor.py --inspect path/to/file.pb")
        print("\nStep 2: Extract a full day of Scotland delays:")
        print("  python gtfsrt_extractor.py --gtfsrt_dir ./gtfsrt_20251001/ --output delays_oct1.csv")


if __name__ == '__main__':
    main()
