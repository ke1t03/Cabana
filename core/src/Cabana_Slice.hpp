/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Slice.hpp
  \brief Slice a single particle property from an AoSoA
*/
#ifndef CABANA_SLICE_HPP
#define CABANA_SLICE_HPP

#include <Cabana_Types.hpp>
#include <Cabana_Utils.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>

//---------------------------------------------------------------------------//
namespace Cabana
{
namespace Impl
{
//! \cond Impl

// A multidimensional View retains the original (SoA, lane, components...)
// ordering when callers ask for Slice::view().
template <class T, std::size_t Rank, int VectorLength>
struct SliceViewDataType;

template <class T, int VectorLength>
struct SliceViewDataType<T, 0, VectorLength>
{
    using type = T* [VectorLength];
};

template <class T, int VectorLength>
struct SliceViewDataType<T, 1, VectorLength>
{
    using type = std::remove_all_extents_t<T>* [VectorLength][std::extent_v<T>];
};

template <class T, int VectorLength>
struct SliceViewDataType<T, 2, VectorLength>
{
    using type = std::remove_all_extents_t<T>*
        [VectorLength][std::extent_v<T, 0>][std::extent_v<T, 1>];
};

template <class T, int VectorLength>
struct SliceViewDataType<T, 3, VectorLength>
{
    using type = std::remove_all_extents_t<T>*
        [VectorLength][std::extent_v<T, 0>][std::extent_v<T, 1>]
        [std::extent_v<T, 2>];
};

template <class T, int VectorLength, int Stride>
struct KokkosViewWrapper
{
    static constexpr auto member_rank = std::rank_v<T>;
    using member_indices = std::make_index_sequence<member_rank>;
    using data_type = typename SliceViewDataType<T, member_rank, VectorLength>::type;
    using cabana_layout = Kokkos::LayoutStride;

    static constexpr std::size_t field_span =
        VectorLength * ( sizeof( T ) / sizeof( std::remove_all_extents_t<T> ) );

    template <std::size_t D>
    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t componentStride()
    {
        if constexpr ( D + 1 == member_rank )
            return 1;
        else
            return std::extent_v<T, D + 1> * componentStride<D + 1>();
    }

    template <std::size_t... D>
    KOKKOS_INLINE_FUNCTION static std::size_t
    extent( const std::size_t num_soa, const std::size_t d,
            std::index_sequence<D...> )
    {
        const std::size_t extents[] = { num_soa, VectorLength,
                                       std::extent_v<T, D>... };
        return d < sizeof...( D ) + 2 ? extents[d] : 0;
    }

    template <std::size_t... D>
    KOKKOS_INLINE_FUNCTION static std::size_t
    stride( const std::size_t d, std::index_sequence<D...> )
    {
        constexpr std::size_t strides[] = {
            Stride, 1, ( VectorLength * componentStride<D>() )... };
        return d < sizeof...( D ) + 2 ? strides[d] : 0;
    }

    KOKKOS_INLINE_FUNCTION static cabana_layout
    createLayout( const std::size_t num_soa )
    {
        cabana_layout layout( num_soa, Stride, VectorLength, 1 );
        for ( std::size_t d = 2; d < member_rank + 2; ++d )
        {
            layout.dimension[d] = extent( num_soa, d, member_indices{} );
            layout.stride[d] = stride( d, member_indices{} );
        }
        return layout;
    }
};

//---------------------------------------------------------------------------//

//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief A slice of an array-of-structs-of-arrays with data access to a single
  multidimensional member.
*/
//---------------------------------------------------------------------------//
template <typename DataType, typename MemorySpace, typename MemoryAccessType,
          int VectorLength, int Stride>
class Slice
{
    using storage_view = Kokkos::View<
        std::remove_all_extents_t<DataType>*, Kokkos::LayoutRight, MemorySpace,
        typename MemoryAccessType::kokkos_memory_traits>;

  public:
    // Ensure the vector length is valid.
    static_assert( Impl::IsVectorLengthValid<VectorLength>::value,
                   "Vector length must be valid" );

    //! Slice type.
    using slice_type =
        Slice<DataType, MemorySpace, MemoryAccessType, VectorLength, Stride>;

    //! Kokkos memory space.
    using memory_space = MemorySpace;
    static_assert( Kokkos::is_memory_space<MemorySpace>() );

    //! Default execution space.
    using execution_space = typename memory_space::execution_space;

    static_assert( is_memory_access_tag<MemoryAccessType>::value,
                   "Slice memory access type must be a Cabana access type" );
    //! Memory access type.
    using memory_access_type = MemoryAccessType;

    //! Member type, including its array dimensions.
    using member_data_type = DataType;

    //! Vector length.
    static constexpr int vector_length = VectorLength;

    //! SoA stride.
    static constexpr int soa_stride = Stride;

    //! Memory space size type.
    using size_type = typename memory_space::size_type;

    //! Index type.
    using index_type = Impl::Index<vector_length>;

    //! Maximum supported rank.
    static constexpr std::size_t max_supported_rank = 3;

    //! Maximum label length.
    static constexpr std::size_t max_label_length = 128;

    //! Kokkos view wrapper.
    using view_wrapper =
        Impl::KokkosViewWrapper<DataType, vector_length, soa_stride>;

    //! Kokkos view type.
    using kokkos_view =
        Kokkos::View<typename view_wrapper::data_type,
                     typename view_wrapper::cabana_layout, MemorySpace,
                     typename MemoryAccessType::kokkos_memory_traits>;

    //! View reference type alias.
    using reference_type = typename storage_view::reference_type;
    //! View value type alias.
    using value_type = typename storage_view::value_type;
    //! View pointer type alias.
    using pointer_type = typename storage_view::pointer_type;
    //! View array layout type alias.
    using view_layout = typename kokkos_view::array_layout;

    //! Default memory access slice type.
    using default_access_slice =
        Slice<DataType, MemorySpace, DefaultAccessMemory, VectorLength, Stride>;
    //! Atomic memory access slice type.
    using atomic_access_slice =
        Slice<DataType, MemorySpace, AtomicAccessMemory, VectorLength, Stride>;
    //! Random memory access slice type.
    using random_access_slice =
        Slice<DataType, MemorySpace, RandomAccessMemory, VectorLength, Stride>;

    // Declare slices of different memory access types to be friends.
    friend class Slice<DataType, MemorySpace, DefaultAccessMemory, VectorLength,
                       Stride>;
    friend class Slice<DataType, MemorySpace, AtomicAccessMemory, VectorLength,
                       Stride>;
    friend class Slice<DataType, MemorySpace, RandomAccessMemory, VectorLength,
                       Stride>;

    // Equivalent Kokkos view rank. This rank assumes that the struct and
    // array dimension are merged. For the true rank of the raw AoSoA data use
    // viewRank(), which accounts for the extra dimension
    // from separate struct and array indices. This enumeration is for
    // compatibility with Kokkos views.
    enum
    {
        rank = std::rank<DataType>::value + 1
    };

  public:
    /*!
      \brief Default constructor.
    */
    Slice() = default;

    /*!
      \brief Constructor.
      \param data Pointer to the first member of the slice.
      \param size The number of tuples in the slice.
      \param num_soa The number of structs in the slice.
      \param label An optional label for the slice.
    */
    Slice( const pointer_type data, const size_type size,
           const size_type num_soa, const std::string& label = "" )
        : _storage( data, num_soa == 0
                              ? 0
                              : ( std::size_t( num_soa ) - 1 ) * soa_stride +
                                    view_wrapper::field_span )
        , _size( size )
        , _num_soa( num_soa )
    {
        std::strncpy( _label, label.c_str(), max_label_length - 1 );
    }

    /*!
      \brief Shallow copy constructor for different memory spaces for
      assigning new memory access traits to the view.
      \tparam MAT Memory access type.
      \param rhs The slice to shallow copy with a potentially different memory
      space.
    */
    template <class MAT>
    Slice( const Slice<DataType, MemorySpace, MAT, VectorLength, Stride>& rhs )
        : _storage( rhs._storage )
        , _size( rhs._size )
        , _num_soa( rhs._num_soa )
    {
        std::strcpy( _label, rhs._label );
    }

    /*!
      \brief Assignment operator for different memory spaces for assigning
      new memory access traits to the view.
      \tparam MAT Memory access type
      \param rhs The slice to shallow copy with a potentially different memory
      space.
      \return A reference to a new slice with a potentially different memory
      space.
     */
    template <class MAT>
    Slice& operator=(
        const Slice<DataType, MemorySpace, MAT, VectorLength, Stride>& rhs )
    {
        _storage = rhs._storage;
        _size = rhs._size;
        _num_soa = rhs._num_soa;
        std::strcpy( _label, rhs._label );
        return *this;
    }

    /*!
      \brief Returns the data structure label.

      \return A string identifying the data structure.
    */
    std::string label() const { return std::string( _label ); }

    /*!
      \brief Returns the total number tuples in the slice.
      \return The number of tuples in the slice.
    */
    KOKKOS_INLINE_FUNCTION
    size_type size() const { return _size; }

    /*!
      \brief Get the number of structs-of-arrays in the container.
      \return The number of structs-of-arrays in the container.
    */
    KOKKOS_INLINE_FUNCTION
    size_type numSoA() const { return _num_soa; }

    /*!
      \brief Get the size of the data array at a given struct member index.
      \param s The struct index to get the array size for.
      \return The size of the array at the given struct index.
    */
    KOKKOS_INLINE_FUNCTION
    size_type arraySize( const size_type s ) const
    {
        if ( s >= _num_soa )
            return 0;
        if ( s + 1 < _num_soa )
            return vector_length;
        const size_type rem = _size % vector_length;
        return ( rem == 0 && _size > 0 ) ? vector_length : rem;
    }

    // ------------
    // 2-D accessor

    //! 2d access for  Rank 0
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 0 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a ) const
    {
        return accessImpl( s, a, typename view_wrapper::member_indices{} );
    }

    //! 2d access for  Rank 1
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 1 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a, const size_type d0 ) const
    {
        return accessImpl( s, a, typename view_wrapper::member_indices{}, d0 );
    }

    //! 2d access for  Rank 2
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 2 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a, const size_type d0,
                const size_type d1 ) const
    {
        return accessImpl( s, a, typename view_wrapper::member_indices{}, d0,
                           d1 );
    }

    //! 2d access for  Rank 3
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 3 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a, const size_type d0,
                const size_type d1, const size_type d2 ) const
    {
        return accessImpl( s, a, typename view_wrapper::member_indices{}, d0,
                           d1, d2 );
    }

    // ------------
    // 1-D accessor

    //! 1d access for  Rank 0
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 0 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i ) const
    {
        return access( index_type::s( i ), index_type::a( i ) );
    }

    //! 1d access for  Rank 1
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 1 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i, const size_type d0 ) const
    {
        return access( index_type::s( i ), index_type::a( i ), d0 );
    }

    //! 1d access for  Rank 2
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 2 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i, const size_type d0,
                    const size_type d1 ) const
    {
        return access( index_type::s( i ), index_type::a( i ), d0, d1 );
    }

    //! 1d access for  Rank 3
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 3 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i, const size_type d0, const size_type d1,
                    const size_type d2 ) const
    {
        return access( index_type::s( i ), index_type::a( i ), d0, d1, d2 );
    }

    // -------------------------------
    // Raw data access.

    /*!
      \brief Get a raw pointer to the data for this member
      \return A raw pointer to the data for this slice.
    */
    KOKKOS_INLINE_FUNCTION
    pointer_type data() const { return _storage.data(); }

    /*!
      \brief Get the rank of the raw data for this slice. This includes
      the struct dimension, array dimension, and all tuple slice
      dimensions.
      \return The rank of the data for this slice.
    */
    KOKKOS_INLINE_FUNCTION
    constexpr size_type viewRank() const { return rank + 1; }

    /*!
      \brief Get the extent of a given raw slice data dimension. This includes
      the struct dimension, array dimension, and all tuple slice
      dimensions.
      \param d The member data dimension to get the extent for.
      \return The extent of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    size_type extent( const size_type d ) const
    {
        return view_wrapper::extent( _num_soa, d,
                                     typename view_wrapper::member_indices{} );
    }

    /*!
      \brief Get the stride of a given raw slice dimension. This includes the
      struct dimension, array dimension, and all tuple slice dimensions.
      \param d The member data dimension to get the stride for.
      \return The stride of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    size_type stride( const size_type d ) const
    {
        return view_wrapper::stride( d, typename view_wrapper::member_indices{} );
    }

    /*!
      \brief Expose the slice as a multidimensional unmanaged Kokkos View.
    */
    KOKKOS_INLINE_FUNCTION
    kokkos_view view() const
    {
        return kokkos_view( data(), view_wrapper::createLayout( _num_soa ) );
    }

  private:
    template <std::size_t... D, class... Indices>
    KOKKOS_FORCEINLINE_FUNCTION reference_type
    accessImpl( const size_type s, const size_type a, std::index_sequence<D...>,
                const Indices... indices ) const
    {
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
        if ( s >= _num_soa || a >= vector_length ||
             ( ( indices >= std::extent_v<DataType, D> ) || ... ) )
            Kokkos::abort( "Cabana::Slice: out of bounds access" );
#endif
        // The compiler knows the strides independently of the View metadata.
        const auto component =
            ( std::size_t( 0 ) + ... +
              ( std::size_t( indices ) *
                view_wrapper::template componentStride<D>() ) );
        return _storage( std::size_t( s ) * soa_stride + a +
                         vector_length * component );
    }

    storage_view _storage;
    size_type _size = 0;
    size_type _num_soa = 0;
    char _label[max_label_length] = {};
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_slice_impl : public std::false_type
{
};

// True only if the type is a member slice *AND* the member slice is templated
// on an AoSoA type.
template <typename DataType, typename MemorySpace, typename MemoryAccessType,
          int VectorLength, int Stride>
struct is_slice_impl<
    Slice<DataType, MemorySpace, MemoryAccessType, VectorLength, Stride>>
    : public std::true_type
{
};
//! \endcond

//! Slice static type checker.
template <class T>
struct is_slice : public is_slice_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Copy to View.
//---------------------------------------------------------------------------//

//! Copy from slice to View. Rank-0
template <class ExecutionSpace, class ViewType, class SliceType>
void copySliceToView(
    ExecutionSpace exec_space, ViewType& view, const SliceType& slice,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        2 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::Rank0",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) { view( i - begin ) = slice( i ); } );
}

//! Copy from slice to View. Rank-1
template <class ExecutionSpace, class ViewType, class SliceType>
void copySliceToView(
    ExecutionSpace exec_space, ViewType& view, const SliceType& slice,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        3 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::Rank1",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                view( i - begin, d0 ) = slice( i, d0 );
        } );
}

//! Copy from slice to View. Rank-2
template <class ExecutionSpace, class ViewType, class SliceType>
void copySliceToView(
    ExecutionSpace exec_space, ViewType& view, const SliceType& slice,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        4 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::Rank2",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                for ( std::size_t d1 = 0; d1 < slice.extent( 3 ); ++d1 )
                    view( i - begin, d0, d1 ) = slice( i, d0, d1 );
        } );
}

//! Copy from slice to View with default execution space.
template <class ViewType, class SliceType>
void copySliceToView( ViewType& view, const SliceType& slice,
                      const std::size_t begin, const std::size_t end )
{
    using exec_space = typename SliceType::execution_space;
    copySliceToView( exec_space{}, view, slice, begin, end );
}

//---------------------------------------------------------------------------//
// Copy from View.
//---------------------------------------------------------------------------//

//! Copy from View to slice. Rank-0
template <class ExecutionSpace, class SliceType, class ViewType>
void copyViewToSlice(
    ExecutionSpace exec_space, SliceType& slice, const ViewType& view,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        2 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copyViewToSlice::Rank0",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) { slice( i - begin ) = view( i ); } );
}

//! Copy from View to slice. Rank-1
template <class ExecutionSpace, class SliceType, class ViewType>
void copyViewToSlice(
    ExecutionSpace exec_space, SliceType& slice, const ViewType& view,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        3 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copyViewToSlice::Rank1",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                slice( i - begin, d0 ) = view( i, d0 );
        } );
}

//! Copy from View to slice. Rank-2
template <class ExecutionSpace, class SliceType, class ViewType>
void copyViewToSlice(
    ExecutionSpace exec_space, SliceType& slice, const ViewType& view,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        4 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copyViewToSlice::Rank2",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                for ( std::size_t d1 = 0; d1 < slice.extent( 3 ); ++d1 )
                    slice( i - begin, d0, d1 ) = view( i, d0, d1 );
        } );
}

//! Copy from View to slice with default execution space.
template <class ViewType, class SliceType>
void copyViewToSlice( SliceType& slice, const ViewType& view,
                      const std::size_t begin, const std::size_t end )
{
    using exec_space = typename SliceType::execution_space;
    copyViewToSlice( exec_space{}, slice, view, begin, end );
}

//! Check slice size (differs from Kokkos View).
template <class SliceType>
void checkSize(
    [[maybe_unused]] SliceType slice, [[maybe_unused]] const std::size_t size,
    typename std::enable_if<is_slice<SliceType>::value, int>::type* = 0 )
{
    // Allow unused for release builds.
    assert( slice.size() == size );
}

//! Check View size (differs from Slice).
template <class ViewType>
void checkSize(
    [[maybe_unused]] ViewType view, [[maybe_unused]] const std::size_t size,
    typename std::enable_if<Kokkos::is_view<ViewType>::value, int>::type* = 0 )
{
    // Allow unused for release builds.
    assert( view.extent( 0 ) == size );
}

//---------------------------------------------------------------------------//

//! Check slice size (differs from Kokkos View).
template <class SliceType>
KOKKOS_INLINE_FUNCTION auto
size( SliceType slice,
      typename std::enable_if<is_slice<SliceType>::value, int>::type* = 0 )
{
    return slice.size();
}

//! Check View size (differs from Slice).
template <class ViewType>
KOKKOS_INLINE_FUNCTION auto size(
    ViewType view,
    typename std::enable_if<Kokkos::is_view<ViewType>::value, int>::type* = 0 )
{
    return view.extent( 0 );
}

} // end namespace Cabana

//---------------------------------------------------------------------------//

#endif // end CABANA_SLICE_HPP
