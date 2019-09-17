#!/usr/bin/env perl
#
#
# Copyright © 2019 Yiğit Sever <yigit.sever@tedu.edu.tr>
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use strict;
use warnings;
use File::Basename;

my %language_codes = (
    als => "sq",
    bul => "bg",
    ell => "el",
    ita => "it",
    ron => "ro",
    slv => "sl",
);

my ($tab_file, $tab_dir) = @ARGV;

if (not defined $tab_file or not defined $tab_file) {
    die "usage: ./tab_creator.pl <tab_file>";
}

if (not -e $tab_file) {
    die "'$tab_file' does not exist";
}

if (not defined $tab_dir && $tab_dir ne '') {
    $tab_dir = './wordnets/tab_files';
}

open (my $fh, '<', $tab_file) or die "Could not open '$tab_file' $!";

my $filename = basename($tab_file);

my $lang_code;
if ($filename =~ m/wn-data-(\w{3})\.tab/) {
    $lang_code = $1;
}


my $short_lang_code = $language_codes{$lang_code};

my $outfilename = $tab_dir . '/' . $short_lang_code . '.tab';
open (my $out_fh, '>', $outfilename) or die "Could not open '$outfilename', $!";

while (my $row = <$fh>) {
    chomp $row;
    if ($row =~ m/$lang_code:def/) {
        if ($row =~ m/^(\d+)-(\w)\s+$lang_code:def\s*\d\s+(.*)$/) {
            my $offset = $1;
            my $pos = $2;
            my $def = $3;
            print $out_fh "$pos $offset $def\n";
        }
    }
}
